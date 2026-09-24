# Study Notes Assistant verification

FAU ID: **wevans2024**

Verified September 23, 2026 on Windows with Python 3.12.14, Chainlit 2.12.0,
LangChain Core 1.6.4, Chroma 1.5.9, FastEmbed 0.8.1, and openai-codex 0.156.1.

## Automated suite

Command: `.venv/Scripts/python.exe -m unittest discover -s tests -v`

**27 tests passed in 0.314 seconds** (test runner time). The initial sandboxed
run hit Windows permission errors reading its own temporary fixtures; the
approved run outside the sandbox passed. All collections/files used for the
successful run were isolated test fixtures, with cleanup registered.

Coverage includes valid JSON and lazy loading; malformed JSON with filename,
line, and column; missing fields and invalid structures/types; empty notes;
invalid UTF-8; metadata preservation; extensionless TXT/JSON upload paths;
duplicate ingestion; identical text in distinct notes/files; CLI/upload reuse;
no partial ingestion of invalid files; propagation of index read failures;
actual Chroma retrieval with sources; exactly one retrieval per question;
grounding instructions; short excerpts; empty-index responses; Codex sign-in
adapter and read-only sandbox; document inventory; upload progress/errors;
per-file size and batch-count limits; initial upload prompt `.send()`; attached
questions; and real TXT/JSON ingestion through the Chainlit handlers.

Python compilation (`python -m compileall -q hw2 tests`) and `git diff --check`
also passed. Git reported only the repository's usual LF-to-CRLF notices.

## Live local embeddings and signed-in Codex

Command: `.venv/Scripts/python.exe tests/live_smoke.py`

Passed using the cached `BAAI/bge-small-en-v1.5` model, an isolated ephemeral
Chroma collection, and the existing Codex sign-in. The first sandboxed attempt
could not initialize Codex's state files under the user's `.codex` directory;
the approved rerun completed successfully. No persistent study-note index was
changed. The test collection was deleted afterward.

- Initial demo ingestion: **4 new chunks from 3 documents** (one TXT file and
  two notes in the JSON file).
- Repeat ingestion: **0 new chunks**.
- Available entries: `rag_notes.txt / rag_notes`,
  `study_notes.json / Chunking and Overlap`, and
  `study_notes.json / Retrieval-Augmented Generation`.
- Question: **What does chunk overlap help preserve?**
- Actual answer: “Chunk overlap helps preserve context and continuity across
  chunk boundaries, reducing the risk of losing meaning when a concept spans
  adjacent chunks. [1], [2]”
- Retrieved source 1: `study_notes.json`, title **Chunking and Overlap**, note
  index 1. Excerpt: “Chunking breaks large documents into smaller pieces, while
  overlap keeps nearby context at boundaries so continuity is preserved across
  chunks.”
- Retrieved sources 2 and 3: two actual `rag_notes.txt` chunks, title
  **rag_notes**, note index 0, each with an excerpt limited to 180 characters.
- Missing-information question: **What is the final exam date for this course?**
- Actual answer: “The final exam date is not available in the provided notes.”

## Chainlit server

Command: `.venv/Scripts/python.exe tests/server_smoke.py`

Passed: the real server returned **HTTP 200** for the homepage. Its
`/project/settings` endpoint advertised spontaneous uploads enabled, TXT/JSON
extensions only, **5 files**, and **10 MB per file**. The test stopped its server
afterward. The local IDE exported `DEBUG=release`, which Chainlit's CLI rejects;
the test and documented launch commands set `DEBUG=false`.

## Verification scope

Live tests exercised real embedding inference, Chroma retrieval, and Codex answer
generation. Automated upload tests exercise the Python Chainlit handlers and
temporary files with messaging captured by mocks. The HTTP smoke checks startup
and configuration; browser drag-and-drop and visual rendering were not manually
tested. The dependency emitted a `langchain-community` deprecation warning;
the existing local embedding integration was preserved as requested.

Changes are uncommitted. The existing staged work and unrelated files were left
in place.
