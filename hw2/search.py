"""Run a similarity search against the local note index."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / ".chromadb"


def build_vectorstore() -> Chroma:
    """Open Chroma with the same FastEmbed model used during ingestion."""
    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return Chroma(
        embedding_function=embedding,
        persist_directory=str(INDEX_DIR),
    )


def search_db(query: str, limit: int = 3) -> list:
    """Return the top matching note chunks for the given query."""
    vectorstore = build_vectorstore()
    return vectorstore.similarity_search(query, k=limit)


def main() -> None:
    """CLI entry point for similarity searches."""
    parser = argparse.ArgumentParser(description="Search the local study-note index.")
    parser.add_argument("query", help="Question or phrase to search for.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum number of chunks to display.")
    args = parser.parse_args()

    print("Study Notes Assistant")
    print("FAU ID: wevans2024")
    matches = search_db(args.query, limit=args.limit)
    if not matches:
        print(f"No matching notes found for: {args.query}")
        return

    print(f"Query: {args.query}")
    for index, match in enumerate(matches, start=1):
        source = match.metadata.get("source", "unknown")
        print(f"\n[{index}] {source}")
        print(match.page_content.strip())


if __name__ == "__main__":
    main()
