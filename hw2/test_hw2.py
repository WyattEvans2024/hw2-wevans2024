"""Automated tests for the Study Notes Assistant homework.

These tests validate the loader, ingestion deduplication, and prompt wiring while
mocking Codex generation so they do not consume actual ChatGPT allowance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import app
from ingest import ingest_directory
from rag_chain import build_prompt, summarize_sources, answer_question
from study_notes_loader import StudyNotesLoader


def test_valid_json_loader_creates_documents(tmp_path: Path) -> None:
    """A valid study-notes JSON payload must create one document per note."""
    payload = {
        "notes": [
            {"title": "Intro", "content": "This is a valid note."},
            {"title": "More", "content": "Another note."},
        ]
    }
    file_path = tmp_path / "notes.json"
    file_path.write_text(json.dumps(payload), encoding="utf-8")

    documents = StudyNotesLoader(file_path).load()

    assert len(documents) == 2
    assert documents[0].metadata["source"] == "notes.json"
    assert documents[0].metadata["title"] == "Intro"
    assert documents[0].metadata["note_index"] == 0
    assert documents[1].page_content.startswith("Another")


def test_malformed_json_raises_clear_value_error(tmp_path: Path) -> None:
    """Malformed JSON should raise a clear validation error instead of producing documents."""
    file_path = tmp_path / "bad.json"
    file_path.write_text('{"notes": [}', encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        StudyNotesLoader(file_path).load()


def test_duplicate_ingestion_skips_existing_chunks(tmp_path: Path) -> None:
    """Repeated ingestion of the same notes should not add duplicate Chroma chunks."""
    source_dir = tmp_path / "notes"
    source_dir.mkdir()
    (source_dir / "study_notes.txt").write_text("RAG is retrieval augmented generation.", encoding="utf-8")

    first = ingest_directory(source_dir, index_dir=tmp_path / "db_one")
    second = ingest_directory(source_dir, index_dir=tmp_path / "db_one")

    assert first == 1
    assert second == 0


def test_build_prompt_uses_context_and_missing_info_language() -> None:
    """The prompt must require the answer to come from context and acknowledge missing information."""
    prompt = build_prompt("What does RAG do?", "RAG combines retrieval and generation.")

    assert "Use the retrieved study notes as reference data only" in prompt
    assert "Answer the user's question using only the context below" in prompt
    assert "If the answer is not present in the context" in prompt
    assert "RAG combines retrieval and generation." in prompt


def test_answer_question_uses_same_sources_for_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The function should retrieve docs once, then use the same docs for prompt and source display."""
    source_dir = tmp_path / "demo_data"
    source_dir.mkdir()
    (source_dir / "study_notes.txt").write_text(
        "Chunk overlap keeps nearby context at the boundaries between chunks.",
        encoding="utf-8",
    )

    class FakeCoder:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, question: str):
            self.calls += 1
            return [type("Doc", (), {"page_content": "Chunk overlap keeps nearby context at the boundaries between chunks.", "metadata": {"source": "study_notes.txt", "title": "Overlap"}})()]

    fake_store = type("FakeStore", (), {"as_retriever": lambda self, search_kwargs=None: FakeCoder()})()

    import rag_chain

    original = rag_chain.CodexAnswerRunnable.invoke
    calls: list[str] = []

    def fake_invoke(self, input_value, config=None, **kwargs):
        calls.append(str(input_value))
        return "Chunk overlap helps preserve continuity."

    monkeypatch.setattr(rag_chain.CodexAnswerRunnable, "invoke", fake_invoke)
    monkeypatch.setattr(rag_chain, "build_vectorstore", lambda index_dir=None: fake_store)

    result = answer_question("What does chunk overlap help preserve?", index_dir=tmp_path / "db")

    assert result["answer"] == "Chunk overlap helps preserve continuity."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source"] == "study_notes.txt"
    assert result["sources"][0]["title"] == "Overlap"
    assert len(calls) == 1
    assert "Chunk overlap keeps nearby context" in calls[0]

    monkeypatch.setattr(rag_chain.CodexAnswerRunnable, "invoke", original)


@pytest.mark.asyncio
async def test_on_message_ingests_attachments_before_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Message attachments should be ingested before question retrieval/search occurs."""
    attachment = tmp_path / "fictional_note.txt"
    attachment.write_text("Fictional note: blue whales are marine mammals.", encoding="utf-8")

    class FakeFile:
        def __init__(self, path: Path, name: str) -> None:
            self.path = str(path)
            self.name = name

    class FakeMessage:
        def __init__(self) -> None:
            self.content = "What are blue whales?"
            self.elements = [FakeFile(attachment, attachment.name)]

    order: list[str] = []

    async def fake_upload_and_ingest(files):
        order.append("ingest")
        assert files[0].name == "fictional_note.txt"
        return [Path(files[0].path)]

    def fake_answer_question(question: str):
        order.append("answer")
        assert question == "What are blue whales?"
        return {
            "answer": "Blue whales are marine mammals.",
            "sources": [{"source": "fictional_note.txt", "title": "Fictional note", "excerpt": "Blue whales are marine mammals."}],
        }

    class FakeChainlitMessage:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def send(self, *args, **kwargs):
            return None

    monkeypatch.setattr(app, "upload_and_ingest", fake_upload_and_ingest)
    monkeypatch.setattr(app, "answer_question", fake_answer_question)
    monkeypatch.setattr(app.cl, "Message", FakeChainlitMessage)

    await app.on_message(FakeMessage())

    assert order == ["ingest", "answer"]


@pytest.mark.asyncio
async def test_upload_and_ingest_reads_uploaded_file_path_and_keeps_original_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Uploaded files should be copied from their local path and keep the original name for ingestion."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    fixture_path = tmp_path / "fictional_note.txt"
    fixture_path.write_text("Fictional note text.", encoding="utf-8")

    class FakeUpload:
        def __init__(self, path: Path, name: str) -> None:
            self.path = str(path)
            self.name = name

    def fake_ingest_directory(directory: Path) -> int:
        assert Path(directory) == upload_dir
        return 1

    monkeypatch.setattr(app, "UPLOADS_DIR", upload_dir)
    monkeypatch.setattr(app, "ingest_directory", fake_ingest_directory)
    monkeypatch.setattr(app, "list_available_documents", lambda: [{"source": "fictional_note.txt", "title": "Fictional note"}])

    class FakeChainlitMessage:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        async def send(self, *args, **kwargs):
            return None

    monkeypatch.setattr(app.cl, "Message", FakeChainlitMessage)

    result = await app.upload_and_ingest([FakeUpload(fixture_path, "fictional_note.txt")])

    assert result == [upload_dir / "fictional_note.txt"]
    assert (upload_dir / "fictional_note.txt").read_text(encoding="utf-8") == "Fictional note text."


def test_summarize_sources_shortens_long_excerpt() -> None:
    """Long excerpts should be shortened for display while keeping the source metadata."""
    docs = [
        type(
            "Doc",
            (),
            {
                "page_content": "A " * 100,
                "metadata": {"source": "example.txt", "title": "Example title"},
            },
        )()
    ]
    sources = summarize_sources(docs)

    assert sources[0]["source"] == "example.txt"
    assert sources[0]["title"] == "Example title"
    assert sources[0]["excerpt"].endswith("...")
