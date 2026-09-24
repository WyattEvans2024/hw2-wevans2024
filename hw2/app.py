"""Chainlit interface for the Study Notes Assistant."""

from __future__ import annotations

from pathlib import Path

import chainlit as cl

from ingest import ingest_directory
from rag_chain import answer_question, list_available_documents

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)


async def upload_and_ingest(files: list[cl.UploadFile]) -> None:
    """Save uploaded TXT/JSON files, ingest them, and confirm the index update."""
    if not files:
        return

    await cl.Message(content="Uploading files and indexing them into the study-note database...").send()
    for upload in files:
        file_path = UPLOADS_DIR / upload.name
        file_path.write_bytes(upload.content)

    ingest_directory(UPLOADS_DIR)
    available = list_available_documents()
    doc_names = ", ".join(f"{item['source']} ({item['title']})" for item in available[:5]) or "No documents indexed yet."
    await cl.Message(content=f"Available documents: {doc_names}").send()


@cl.on_chat_start
async def on_chat_start() -> None:
    """Display the app banner and ask for initial TXT or JSON uploads."""
    welcome_text = (
        "**Study Notes Assistant**\n\n"
        "FAU ID: wevans2024\n\n"
        "Upload TXT or JSON study notes to index them."
    )
    await cl.Message(content=welcome_text).send()

    available = list_available_documents()
    if available:
        docs_text = "\n".join(f"- {entry['source']} — {entry['title']}" for entry in available)
        await cl.Message(content=f"Available documents:\n{docs_text}").send()

    uploaded = await cl.AskFileMessage(
        content="Upload one or more TXT or JSON study-note files (up to 10 MB total).",
        accept={"text/plain": [".txt"], "application/json": [".json"]},
        max_size_mb=10,
        max_files=5,
    )
    if uploaded:
        await upload_and_ingest(uploaded)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer the question from the retrieved notes and display the source metadata."""
    user_query = message.content.strip()
    if not user_query:
        return

    await cl.Message(content="Retrieving relevant study notes and generating an answer...").send()
    result = answer_question(user_query)
    answer = str(result["answer"])
    await cl.Message(content=answer).send()

    sources = result.get("sources", [])
    if sources:
        source_lines = ["**Sources**"]
        for source in sources:
            source_lines.append(
                f"- {source['source']} — {source['title']}: {source['excerpt']}"
            )
        await cl.Message(content="\n".join(source_lines)).send()


if __name__ == "__main__":
    cl.run(host="127.0.0.1", port=8000)
