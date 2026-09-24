"""Chainlit interface for the Study Notes Assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chainlit as cl

from ingest import ingest_directory
from rag_chain import answer_question, list_available_documents

UPLOADS_DIR = Path(__file__).resolve().parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)
SUPPORTED_EXTENSIONS = {".txt", ".json"}


def _coerce_uploaded_file(upload: Any) -> tuple[str, Path]:
    """Normalize a Chainlit upload object or file response to a real local file path."""
    if isinstance(upload, dict):
        name = upload.get("name")
        path_value = upload.get("path")
    else:
        name = getattr(upload, "name", None)
        path_value = getattr(upload, "path", None)

    if not name:
        raise ValueError("Uploaded file is missing a filename.")

    safe_name = Path(name).name
    if not safe_name:
        raise ValueError("Uploaded file has an empty filename.")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported uploaded file type: {safe_name}. Supported types: .txt, .json")

    source_path = Path(path_value) if path_value else None
    if source_path is None:
        content = getattr(upload, "content", None)
        if content is None and isinstance(upload, dict):
            content = upload.get("content")
        if content is None:
            raise ValueError(f"Uploaded file {safe_name} has no path or content to read.")
        destination = UPLOADS_DIR / safe_name
        destination.write_bytes(content if isinstance(content, bytes) else str(content).encode("utf-8"))
        return safe_name, destination

    if not source_path.exists():
        raise FileNotFoundError(f"Uploaded file not found on disk: {source_path}")

    target_path = UPLOADS_DIR / safe_name
    target_path.write_bytes(source_path.read_bytes())
    return safe_name, target_path


async def upload_and_ingest(files: list[Any]) -> list[Path]:
    """Copy uploaded TXT/JSON files to the app upload directory and ingest them into Chroma."""
    if not files:
        return []

    await cl.Message(content="Uploading files and indexing them into the study-note database...").send()

    copied_paths: list[Path] = []
    for upload in files:
        _, destination = _coerce_uploaded_file(upload)
        copied_paths.append(destination)

    try:
        count = ingest_directory(UPLOADS_DIR)
        available = list_available_documents()
        doc_names = ", ".join(f"{item['source']} ({item['title']})" for item in available[:5]) or "No documents indexed yet."
        await cl.Message(content=f"Indexing complete. {count} new chunks added. Available documents: {doc_names}").send()
        return copied_paths
    except Exception as exc:  # pragma: no cover - runtime feedback path
        await cl.Message(content=f"Upload indexing failed: {exc}").send()
        raise


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
    ).send()
    if uploaded:
        await upload_and_ingest(uploaded)


def _extract_attachments(message: cl.Message) -> list[Any]:
    """Return uploaded file elements from a user message, if any."""
    if not getattr(message, "elements", None):
        return []

    attachments: list[Any] = []
    for element in message.elements:
        if getattr(element, "path", None):
            attachments.append(element)
    return attachments


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Ingest any attached study notes before answering the question and display retrieved sources."""
    attachments = _extract_attachments(message)
    if attachments:
        await upload_and_ingest(attachments)

    user_query = (message.content or "").strip()
    if not user_query:
        if attachments:
            await cl.Message(content="Files were indexed. Ask a question to query them.").send()
        return

    await cl.Message(content="Retrieving relevant study notes and generating an answer...").send()
    result = answer_question(user_query)
    answer = str(result["answer"])
    await cl.Message(content=answer).send()

    sources = result.get("sources", [])
    if sources:
        source_lines = ["**Sources**"]
        for source in sources:
            source_lines.append(f"- {source['source']} — {source['title']}: {source['excerpt']}")
        await cl.Message(content="\n".join(source_lines)).send()


if __name__ == "__main__":
    cl.run(host="127.0.0.1", port=8000)
