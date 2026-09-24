"""Opt-in smoke test using cached local FastEmbed and the existing Codex sign-in.

Run from the repo root with ``.venv/Scripts/python.exe tests/live_smoke.py``.
This calls Codex twice and leaves the user's persistent Chroma index untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hw2"))

from ingest import BASE_DIR, ingest_directory
from rag_chain import answer_question, list_available_documents


def main() -> None:
    """Exercise real embeddings, duplicate ingestion, and signed-in generation."""
    # Deliberately test the already-cached model without downloading weights.
    os.environ["HF_HUB_OFFLINE"] = "1"
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    store = Chroma(
        client=chromadb.EphemeralClient(),
        collection_name=f"study-notes-smoke-{uuid4().hex}",
        embedding_function=embeddings,
    )
    try:
        first = ingest_directory(BASE_DIR / "demo_data", vectorstore=store)
        second = ingest_directory(BASE_DIR / "demo_data", vectorstore=store)
        assert first > 0 and second == 0, (first, second)
        print(json.dumps({
            "new_chunks": first,
            "duplicate_chunks": second,
            "available": list_available_documents(vectorstore=store),
        }, indent=2), flush=True)
        for question in (
            "What does chunk overlap help preserve?",
            "What is the final exam date for this course?",
        ):
            result = answer_question(question, vectorstore=store)
            assert result["answer"] and result["sources"]
            print(json.dumps({"question": question, **result}, indent=2), flush=True)
    finally:
        store.delete_collection()


if __name__ == "__main__":
    main()
