# ProductionGradeRAGPythonApp

This repository is a small local Retrieval-Augmented Generation (RAG) app built around:

- `Streamlit` for the UI
- `FastAPI` for hosting Inngest functions
- `Inngest` for event-driven ingestion and query workflows
- `OpenAI` for embeddings and answer generation
- `Qdrant` for vector storage and retrieval
- `LlamaIndex` for PDF reading and chunking

Despite the name, the current codebase is closer to a compact local prototype than a full production-grade system. The README below documents the app as it exists today.

## Provider support

The app now supports provider selection for answer generation:

- `openai` for embeddings and answer generation
- `groq` for answer generation

For embeddings, the app now supports:

- `openai`
- `google`

Groq's official docs currently expose OpenAI-compatible text generation, but do not list embedding models or an embeddings API on the pages used for this update, so `groq` is still not supported for embeddings.

For this codebase's text-only PDF RAG flow, the cheapest Google embedding model is:

- `gemini-embedding-001`

Google's official Gemini API pricing page lists:

- `gemini-embedding-001`: `$0.15` per 1M input tokens on standard pricing
- `gemini-embedding-2`: `$0.20` per 1M text input tokens on standard pricing

So `gemini-embedding-001` is now the default Google embedding model in this app.

## Execution modes

The app can now run in two modes:

- Direct mode: Streamlit calls the RAG logic directly with no Inngest server required
- Inngest mode: Streamlit sends events and the FastAPI app handles them through Inngest

For the simplest local setup, use direct mode:

```env
USE_INNGEST=false
```

## What the app does

The app supports two flows:

1. Upload a PDF in the Streamlit UI.
2. The UI saves the file to `uploads/` and sends an Inngest event named `rag/ingest_pdf`.
3. A FastAPI-hosted Inngest function:
   - reads the PDF
   - splits it into text chunks
   - creates embeddings with `text-embedding-3-large`
   - upserts the chunks into a Qdrant collection named `docs`

Then:

1. Enter a question in the Streamlit UI.
2. The UI sends an Inngest event named `rag/query_pdf_ai`.
3. A second Inngest function:
   - embeds the question
   - retrieves similar chunks from Qdrant
   - sends the retrieved context to the configured LLM provider
   - returns a concise answer plus source names

## Code structure

- [streamlit_app.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/streamlit_app.py) contains the user interface for PDF upload and Q&A.
- [main.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/main.py) defines the FastAPI app and both Inngest functions.
- [data_loader.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/data_loader.py) loads PDFs, chunks text, and calls the OpenAI embeddings API.
- [vector_db.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/vector_db.py) wraps Qdrant collection creation, upsert, and search.
- [custom_types.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/custom_types.py) defines the Pydantic models passed between workflow steps.
- [provider_config.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/provider_config.py) centralizes provider, model, API key, and embedding-dimension configuration.

## Runtime architecture

The current app expects three moving pieces during local development:

- A FastAPI server for the Inngest handlers
- A Streamlit server for the UI
- A Qdrant server reachable at `http://localhost:6333`

In addition, the query UI polls the Inngest API at:

- `http://127.0.0.1:8288/v1` by default

That base URL can be overridden with `INNGEST_API_BASE`.

## Requirements

- Python `>=3.13` according to [pyproject.toml](/Users/maryam/Projects/ProductionGradeRAGPythonApp/pyproject.toml)
- An `OPENAI_API_KEY`
- A `GROQ_API_KEY` if `LLM_PROVIDER=groq`
- A `GEMINI_API_KEY` if `EMBEDDING_PROVIDER=google`
- A running Qdrant instance on `localhost:6333`, unless you change `vector_db.py`
- An Inngest development or server environment so `client.send(...)` has somewhere to deliver events and the UI can poll run output

## Install dependencies

This repo includes a `uv.lock`, but `uv` is not required in principle. Use either `uv` or `pip`.

### Option 1: `uv`

```bash
uv sync
```

### Option 2: `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environment

Create a `.env` file in the project root with at least:

```env
USE_INNGEST=false
INNGEST_DEV=1
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

If you want Google embeddings, set:

```env
EMBEDDING_PROVIDER=google
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_BASE=https://generativelanguage.googleapis.com/v1beta
GOOGLE_EMBED_BATCH_SIZE=8
GOOGLE_EMBED_DELAY_SECONDS=1.0
GOOGLE_EMBED_MAX_RETRIES=8
```

If you want Groq for answer generation, set:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_LLM_MODEL=llama-3.3-70b-versatile
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

If your Inngest setup needs additional environment variables for event delivery, add them as required by your local Inngest configuration.

## Run locally

### Simplest path: no Inngest

If you just want the app running locally, you only need:

1. Qdrant
2. A valid `OPENAI_API_KEY`
3. A valid `GROQ_API_KEY` only if `LLM_PROVIDER=groq`
4. Streamlit

Set:

```env
USE_INNGEST=false
```

Then run:

```bash
python3 -m streamlit run streamlit_app.py
```

In this mode, the Streamlit app ingests PDFs and answers questions directly without the FastAPI app or Inngest dev server.

### 1. Start Qdrant

You need a Qdrant server listening on `http://localhost:6333`.

One common way is Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 2. Start the FastAPI app

```bash
python3 -m uvicorn main:app --reload --port 8000
```

This app hosts the two Inngest functions:

- `RAG: Ingest PDF`
- `RAG: Query PDF`

### 3. Start your Inngest dev server

You need an Inngest dev/runtime process that can:

- receive events sent by the Streamlit app
- invoke the FastAPI-served functions
- expose run status/output for polling

The Streamlit app expects the Inngest API to be available at `http://127.0.0.1:8288/v1` unless overridden.

### 4. Start the Streamlit UI

```bash
python3 -m streamlit run streamlit_app.py
```

Then:

1. Upload a PDF.
2. Wait for the ingestion event to complete.
3. Ask a question about the ingested content.

## Data flow details

### Ingestion flow

- PDF text is loaded with `PDFReader`
- Text is split with `SentenceSplitter(chunk_size=1000, chunk_overlap=200)`
- Embeddings are generated with `text-embedding-3-large`
- Chunk IDs are deterministic UUIDv5 values derived from `source_id` and chunk index
- Qdrant payloads store:
  - `source`
  - `text`

### Query flow

- The question is embedded with the same embedding model
- Top `k` chunks are retrieved from Qdrant
- The retrieved chunk text is concatenated into a context block
- The configured `LLM_PROVIDER` model generates the final answer
- The result returned to the UI includes:
  - `answer`
  - `sources`
  - `num_contexts`

## Important implementation notes

### 1. `qdrant_storage/` is not the active storage path

This repository contains a checked-in `qdrant_storage/` directory, but the current code does not use it directly.

[vector_db.py](/Users/maryam/Projects/ProductionGradeRAGPythonApp/vector_db.py) connects to Qdrant over HTTP:

- default URL: `http://localhost:6333`
- default collection: `docs`

So the checked-in `qdrant_storage/` folder should be treated as repository data or an old local artifact, not as the active runtime configuration.

### 2. Uploads are stored on disk

Uploaded PDFs are written to:

- `uploads/<filename>.pdf`

That directory is created automatically by the Streamlit app.

### 3. The current answer prompt is strict

The LLM is instructed to answer only from the retrieved context:

- system intent: answer using only provided context
- user intent: answer concisely using the retrieved chunks

If retrieval is weak, answer quality will be weak as well.

### 4. Groq is currently generation-only in this app

The current provider implementation supports:

- `LLM_PROVIDER=openai`
- `LLM_PROVIDER=groq`

Embeddings support:

- `EMBEDDING_PROVIDER=openai`
- `EMBEDDING_PROVIDER=google`

The default Google embedding model is:

- `GOOGLE_EMBEDDING_MODEL=gemini-embedding-001`

You can also override it to:

- `GOOGLE_EMBEDDING_MODEL=gemini-embedding-2`

To reduce rate-limit errors during ingestion, the app now batches and paces Google embedding requests with:

- `GOOGLE_EMBED_BATCH_SIZE`
- `GOOGLE_EMBED_DELAY_SECONDS`
- `GOOGLE_EMBED_MAX_RETRIES`

This follows Google's rate-limit guidance that Gemini limits are enforced across RPM, TPM, and RPD, and that Batch API traffic has separate limits.

If you change embedding models, also update:

- `EMBEDDING_DIM`

Otherwise the Qdrant collection dimension can mismatch the vectors being inserted.

### 5. Inngest is now optional for local use

When `USE_INNGEST=false`, the app does not require:

- the FastAPI endpoint
- the Inngest dev server
- event polling

When `USE_INNGEST=true`, the previous event-driven flow remains available.

### 6. There is no auth, multi-user isolation, or document lifecycle management

The current implementation does not yet include:

- authentication or authorization
- per-user document separation
- delete/update flows for documents
- metadata filtering during retrieval
- automated evaluation
- test coverage
- production deployment configuration

## Known gaps

Based on the current code, these are the main practical limitations:

- The app depends on several local services being started manually.
- The README had been empty, so operational setup was undocumented.
- There are no tests in the repository.
- The code assumes a working Inngest runtime, but does not include scripts for launching it.
- The Qdrant connection is hard-coded in `vector_db.py`.
- Groq is only wired for answer generation, not embeddings.
- The project name says "production grade", but the implementation is still a simple prototype.

## Suggested next improvements

If you want to evolve this into a stronger project, the most useful next steps would be:

1. Add startup scripts or `make` targets for the common local workflows.
2. Add a `docker-compose.yml` for Qdrant and the app services.
3. Make Qdrant URL, collection name, and model names configurable.
4. Add tests for chunking, vector upsert, and query behavior.
5. Add document deletion and source-level re-ingestion.
6. Add retrieval metadata and filtering.
7. Add startup scripts for the full local developer workflow.

## Quick summary

This codebase is a local event-driven PDF RAG app:

- Streamlit uploads and queries
- Inngest orchestrates the workflows
- Configurable providers handle embeddings and answers
- Qdrant stores and retrieves chunks

It is functional as a prototype, but it still needs operational hardening before the repository name fully matches the implementation.
