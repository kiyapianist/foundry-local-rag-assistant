# Local RAG AI Assistant with Microsoft Foundry Local

[![Tests](https://img.shields.io/badge/tests-16%20passed-success.svg)](#run-tests)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Offline](https://img.shields.io/badge/runtime-offline-success.svg)](#offline-mode)

This project is an offline document Q&A assistant for the internship project.
It uses Retrieval-Augmented Generation (RAG), SQLite, and Microsoft Foundry Local
so questions can be answered from local documents without cloud API calls during runtime.

## What It Does

- Loads local `.txt`, `.pdf`, and `.docx` documents
- Splits documents into searchable chunks
- Generates embeddings with Foundry Local: `qwen3-embedding-0.6b`
- Stores chunks and embeddings in SQLite: `data/knowledge.db`
- Retrieves the most relevant chunks for each question
- Rejects low-confidence retrieval instead of forcing an unsupported answer
- Generates answers with Foundry Local: `phi-3.5-mini`
- Shows answers in a Streamlit web UI or a command-line interface
- Runs offline after the one-time model download

## Project Structure

```text
rag-assistant/
  app.py                    Streamlit web UI
  main.py                   CLI version
  requirements.txt          Python dependencies
  scripts/
    setup_foundry.py        One-time Foundry model download/cache script
    ingest.py               Document ingestion script
    run_app.bat             Windows launcher for the Streamlit UI
    run_app.ps1             PowerShell launcher for the Streamlit UI
  src/
    config.py               Project settings
    foundry_runtime.py      Foundry Local runtime/client setup
    loader.py               TXT, PDF, and DOCX text extraction
    database.py             SQLite storage
    embedder.py             Foundry embedding calls
    retriever.py            Similarity search
    llm.py                  Foundry chat generation
    pipeline.py             End-to-end RAG pipeline
  data/
    documents/              Local knowledge-base files
    foundry/models/         Cached Foundry Local models
    knowledge.db            SQLite database
  tests/
    test_pipeline.py        Basic tests
```

## Setup

Use Python 3.10 or newer.

```powershell
git clone https://github.com/kiyapianist/foundry-local-rag-assistant.git
cd foundry-local-rag-assistant
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## One-Time Foundry Local Model Setup

This step needs internet once. It downloads the Foundry Local models into
`data/foundry/models`.

```powershell
python scripts/setup_foundry.py
```

Models used:

- Chat model: `phi-3.5-mini`
- Embedding model: `qwen3-embedding-0.6b`

After this step, runtime is offline by default.

## Add Documents

Put documents in:

```text
data/documents/
```

Supported formats:

- `.txt`
- `.pdf`
- `.docx`

## Build the Knowledge Base

```powershell
python scripts/ingest.py --yes
```

This reads the documents, chunks them, generates Foundry Local embeddings, and
stores everything in SQLite.

## Run the App

Web UI:

```powershell
.\scripts\run_app.bat
```

CLI:

```powershell
python main.py
```

## Offline Mode

Offline mode is enabled by default through `RAG_OFFLINE_MODE=1`.

To be explicit:

```powershell
$env:RAG_OFFLINE_MODE="1"
streamlit run app.py
```

If you need to download or refresh Foundry models, run setup while online:

```powershell
$env:RAG_OFFLINE_MODE="0"
python scripts/setup_foundry.py
```

## Run Tests

```powershell
python -m pytest tests\ -v
```

## Evaluate Quality

Run the fast retrieval and refusal evaluation:

```powershell
python scripts/evaluate.py --mode retrieval --output evaluation/results-retrieval.json
```

Run the complete answer-quality evaluation (slower because it invokes the local LLM):

```powershell
python scripts/evaluate.py --mode full --output evaluation/results-full.json
```

The evaluation checks expected sources, unsupported-question refusal, answer terms,
and latency. See `docs/PROJECT_REPORT.md` for the architecture, safeguards, and
known limitations.

## Data Privacy

Files under `data/documents/`, model caches, chat/database contents, and runtime
logs stay local and are ignored by Git. Review documents for personal information
before copying or demonstrating the project. Foundry model generation runs offline
after the one-time setup download.

## Limitations

- Image-only/scanned PDFs need OCR before this project can index their text.
- Response speed depends on local CPU/GPU/NPU capability; the first answer includes
  model startup time.
- The SQLite brute-force vector search is intended for a small teaching knowledge base.

## Presentation Notes

Explain the project as:

1. A local RAG assistant that answers from private documents.
2. Foundry Local runs the chat and embedding models on the device.
3. SQLite stores document chunks and vectors locally.
4. Retrieval sends only the most relevant chunks to the model.
5. The app can say it does not know when the answer is not in the documents.

Good demo flow:

1. Show `data/documents`.
2. Run or show `python scripts/ingest.py --yes`.
3. Open the Streamlit UI.
4. Ask one answerable question.
5. Show the retrieved context and source.
6. Ask one question that is not in the documents.

