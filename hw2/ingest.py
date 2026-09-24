"""Ingest TXT and JSON study notes into the local Chroma vector store."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma

from study_notes_loader import StudyNotesLoader

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / ".chromadb"


def build_vectorstore(index_dir: Path | None = None) -> Chroma:
    """Create a local Chroma database using FastEmbed BAAI/bge-small-en-v1.5."""
    target_dir = index_dir or INDEX_DIR
    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma(
        embedding_function=embedding,
        persist_directory=str(target_dir),
    )


def load_txt_documents(directory: Path) -> list:
    """Load all TXT files from a directory into LangChain documents."""
    loader = DirectoryLoader(
        str(directory),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    for document in documents:
        source = Path(document.metadata.get("source", ""))
        try:
            relative_path = source.relative_to(directory)
        except ValueError:
            relative_path = source.name if source.name else "unknown.txt"
        document.metadata["source"] = str(relative_path)
    return documents


def load_json_documents(directory: Path) -> list:
    """Load all JSON study-note files from a directory into LangChain documents."""
    documents: list = []
    for json_file in sorted(directory.rglob("*.json")):
        loader = StudyNotesLoader(json_file)
        file_documents = loader.load()
        for document in file_documents:
            document.metadata["source"] = json_file.name
        documents.extend(file_documents)
    return documents


def load_directory_documents(directory: Path) -> list:
    """Collect TXT and JSON study-note documents from a directory."""
    documents = load_txt_documents(directory)
    documents.extend(load_json_documents(directory))
    return documents


def deduplicate_chunks(chunks: list, vectorstore: Chroma) -> list:
    """Skip chunks that already exist in the persistent index."""
    try:
        existing = vectorstore.get(include=["documents"])
        seen_texts = {item.strip() for item in existing.get("documents", []) if item}
    except Exception:
        seen_texts = set()

    unique_chunks: list = []
    for chunk in chunks:
        content = chunk.page_content.strip()
        if content not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(content)
    return unique_chunks


def ingest_directory(directory: Path, index_dir: Path | None = None) -> int:
    """Split the TXT/JSON notes into chunks and add any new chunks to Chroma."""
    vectorstore = build_vectorstore(index_dir=index_dir)
    documents = load_directory_documents(directory)
    if not documents:
        raise FileNotFoundError(f"No supported study-note files found in: {directory}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    new_chunks = deduplicate_chunks(chunks, vectorstore)

    if new_chunks:
        vectorstore.add_documents(new_chunks)

    print(f"Study Notes Assistant: ingested {len(new_chunks)} new chunks from {len(documents)} document(s).")
    print("FAU ID: wevans2024")
    return len(new_chunks)


def main() -> None:
    """CLI entry point for local study-note ingestion."""
    parser = argparse.ArgumentParser(description="Ingest study notes into the local Chroma index.")
    parser.add_argument("--directory", default="demo_data", help="Directory containing TXT or JSON study notes.")
    args = parser.parse_args()

    target_directory = (BASE_DIR / args.directory).resolve()
    ingest_directory(target_directory)


if __name__ == "__main__":
    main()
