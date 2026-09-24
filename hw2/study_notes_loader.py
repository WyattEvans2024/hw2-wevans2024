"""Custom LangChain loader for JSON study-note files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class StudyNotesLoader(BaseLoader):
    """Load a JSON file of the form {"notes": [{"title": ..., "content": ...}]}."""

    def __init__(self, file_path: str | Path) -> None:
        """Store the JSON file path for later lazy loading."""
        self.file_path = Path(file_path)

    def lazy_load(self) -> Iterator[Document]:
        """Yield a LangChain document for each valid note in the JSON payload."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Study notes file not found: {self.file_path}")

        try:
            text = self.file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Unable to decode {self.file_path.name} as UTF-8 JSON.") from exc

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON in {self.file_path.name}: {exc.msg} at line {exc.lineno}, column {exc.colno}."
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"Study notes JSON in {self.file_path.name} must be an object with a 'notes' list."
            )

        notes = payload.get("notes")
        if not isinstance(notes, list):
            raise ValueError(
                f"Study notes JSON in {self.file_path.name} must contain a 'notes' list."
            )

        for note_index, note in enumerate(notes):
            if not isinstance(note, dict):
                raise ValueError(
                    f"Note at index {note_index} in {self.file_path.name} must be an object with 'title' and 'content'."
                )

            title = note.get("title")
            content = note.get("content")

            if not isinstance(title, str) or not title.strip():
                raise ValueError(
                    f"Note at index {note_index} in {self.file_path.name} is missing a valid 'title' string."
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Note at index {note_index} in {self.file_path.name} is missing a valid 'content' string."
                )

            yield Document(
                page_content=content.strip(),
                metadata={
                    "source": self.file_path.name,
                    "title": title.strip(),
                    "note_index": note_index,
                },
            )
