# Study Notes Assistant

## Features and custom functionality

This homework adapts the official course starter into a local Windows study-note assistant with these custom features:

- JSON note ingestion using a custom `StudyNotesLoader` that validates the schema and yields one `Document` per note.
- TXT ingestion kept alongside JSON ingestion.
- Local Chroma vector store with FastEmbed using `BAAI/bge-small-en-v1.5`.
- Retrieval and answer flow that fetches relevant notes once, then reuses that same context for answer generation and source presentation.
- Source attribution that displays actual filenames, titles, and excerpts alongside each answer.
- Chainlit upload support for TXT and JSON files with a reasonable 10 MB limit and progress feedback.
- Supported Codex ChatGPT sign-in integration with read-only sandbox usage for generation.
- App and CLI commands for both ingestion and retrieval.

## Windows and uv setup

From a fresh checkout, use a shell that has Python 3.12 and `uv` available.

1. Open PowerShell in the repository root.
2. Create the project environment:

   uv sync --python 3.12

3. Activate the environment if you want to run commands directly:

   .\.venv\Scripts\Activate.ps1

4. If PowerShell script execution is blocked, use the command shell with `uv run ...` so the project environment is used without changing system policy.

## Supported ChatGPT/Codex sign-in requirements

The application uses the official OpenAI Codex Python SDK and the supported ChatGPT sign-in path. The app requires a valid local Codex session authenticated with ChatGPT or an allowed Codex login. It does not use API-key billing or provider fallback.

Authentication requirements:

- Sign in with ChatGPT through the Codex CLI or desktop app using the official browser flow.
- Do not expose credentials in code, logs, or shell history.
- Use the default Codex login cache; do not change global Codex settings.
- The generation layer uses `Sandbox.read_only` to keep answer execution restricted to read-only access.

## Initial embedding-model download

The first run of the app or ingestion command downloads the FastEmbed model `BAAI/bge-small-en-v1.5` to the local cache. This is expected and is the initial model download. The model is then reused on subsequent runs.

## Exact commands

From the repo root:

### Ingest demo data

uv run python hw2/ingest.py --directory hw2/demo_data

### Search the local database

uv run python hw2/search.py "What does chunk overlap help preserve?"

### Run tests

uv run pytest hw2/test_hw2.py

### Launch the app

uv run chainlit run hw2/app.py --host 127.0.0.1 --port 8000

## JSON format

The custom JSON loader expects this structure:

```json
{
  "notes": [
    {"title": "Note title", "content": "Note text"},
    {"title": "Another title", "content": "Another note."}
  ]
}
```

The loader validates the object and requires a `notes` list. Each item must be an object with non-empty string values for `title` and `content`.

## Demonstration questions

The demo notes support questions such as:

- What does chunk overlap help preserve?
- What is retrieval-augmented generation?
- Why is source attribution important?

## Actual observed limitations

- The project intentionally uses the local FastEmbed model cache and the local Chroma database, so it is not a cloud-backed setup.
- The Chainlit app requires a valid local Codex login to generate final answers.
- The first ingestion or answer run may download and initialize the embedding model, which adds a small delay on first use.
- The generated app does not rely on the ignored `02_LangChain` folder, the original ZIP, or a previous local database.

## How this derives from the official course starter

This application keeps the course starter’s original principles:

- Recursive character chunking via `RecursiveCharacterTextSplitter`
- Chroma vector storage
- retrieval-based context gathering
- LCEL prompt wiring
- Chainlit UI pattern for user interaction
- local reference material stored as text or JSON notes

The key changes are the use of FastEmbed instead of the Google/Vertex embedding stack, the conversion to a local JSON/TXT-only study-note system, the Codex read-only generation wrapper, and the added source-attribution display.

## Fresh-checkout verification

A fresh checkout should work using only tracked project files and the documented prerequisites. The project does not rely on:

- the ignored `02_LangChain` folder
- any original ZIP or development archive
- an absolute path into a personal machine directory
- an existing database already in the repo

The tracked files include the Python project metadata, lock file, demo documents, and the homework source under `hw2`.
