# Study Notes Assistant

FAU ID: **wevans2024**

Ask questions about local TXT and JSON study notes. FastEmbed
(`BAAI/bge-small-en-v1.5`) computes embeddings locally, Chroma persists the index
under `hw2/.chromadb`, and the existing signed-in Codex SDK generates answers.
The first embedding-model use may download model weights. Existing Codex sign-in
is reused; this app does not replace it with an API-key integration.

## Run

Use Python 3.12 and the dependencies in `pyproject.toml` / `uv.lock`:

```powershell
uv sync --frozen
.venv\Scripts\python.exe hw2\ingest.py --directory demo_data
Set-Location hw2
$env:DEBUG = "false"
..\.venv\Scripts\chainlit.exe run app.py --host 127.0.0.1 --port 8000
```

Launch Chainlit from **`hw2`**, as shown, so it uses
`hw2/.chainlit/config.toml`, including the upload limits. Alternatively, set
`CHAINLIT_APP_ROOT` to the absolute `hw2` directory before starting Chainlit.
The `DEBUG` assignment avoids a Chainlit CLI error if an IDE exports a nonboolean
value such as `release`.

Upload using the initial prompt, `/upload`, or attachments on any chat message.
Both TXT and JSON are accepted, with at most **5 files per batch and 10 MB per
file**. Each file receives indexing progress and a success, duplicate, or error
message. A rejected file does not prevent the rest of the batch from loading.
An attached question is answered after its attachments finish indexing.
Use `/documents` to see all indexed filenames and note titles.

## Custom JSON loader

`hw2/study_notes_loader.py` defines `StudyNotesLoader(BaseLoader)` with
`lazy_load()`. `hw2/demo_data/study_notes.json` provides a two-note demo:

```json
{"notes": [{"title": "Note title", "content": "Note text"}]}
```

The root must be an object containing a `notes` list. Each note must be an object
with nonblank string `title` and `content` fields. An empty list is valid and
produces no documents. UTF-8 and UTF-8 with a BOM are supported. The loader
validates the complete payload before yielding one `Document` per note, keeping
the original content and metadata `source` (filename), `title`, and `note_index`
(zero-based). Errors identify the filename and missing/invalid field or JSON
line and column. TXT remains supported, with its filename stem as the title.

Uploads call the same `ingest_file()` / shared loading, splitting, and indexing
functions used by command-line ingestion. Original filenames survive Chainlit's
temporary upload paths. Exact repeated chunks are skipped while distinct source
files and note indices retain their provenance. Ingestion is additive: changed
notes add new chunks; it does not remove older versions from the index.

Each question retrieves once. That same list supplies the model context and the
displayed filenames, titles, and excerpts (up to 180 characters). Numbered source
entries match the context numbering used for answer citations. The model is
instructed to use only that context, acknowledge missing information, and treat
document text and metadata as reference data rather than instructions. An empty
index returns a missing-information response without calling Codex.

## Tests

From the repository root:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Optional smoke checks:

```powershell
.venv\Scripts\python.exe tests\server_smoke.py
.venv\Scripts\python.exe tests\live_smoke.py
```

The live smoke test requires cached embedding weights and the existing Codex
sign-in; it sends two demo questions to Codex using an isolated Chroma collection.

The automated suite uses temporary files, isolated Chroma collections, deterministic
test embeddings, and mocked answer generation / Chainlit messaging. It checks the
JSON contract, malformed and missing fields, TXT/JSON uploads, duplicate ingestion,
source provenance, one retrieval per question, and upload limits. Production still
uses local FastEmbed and the existing Codex sign-in.

See [TEST_RESULTS.md](TEST_RESULTS.md) for the actual verification results and
any limits on live testing.

API references: [Chainlit file uploads](https://docs.chainlit.io/api-reference/ask/ask-for-file)
and [LangChain BaseLoader](https://reference.langchain.com/python/langchain-core/document_loaders/base/BaseLoader).
