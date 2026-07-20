# Five-Minute Presentation Outline

## 1. Problem (30 seconds)

Cloud assistants can expose private documents and may answer from unsupported outside knowledge. This project provides local document Q&A with no cloud API dependency during runtime.

## 2. Architecture (60 seconds)

Show `data/documents`, then explain: load and chunk documents, embed locally, store vectors in SQLite, retrieve relevant chunks, and send only accepted context to the local chat model.

## 3. Live Demo (2 minutes)

1. Show that both Foundry models report cached and offline-ready.
2. Ask an answerable question and show its source and retrieved context.
3. Ask “What is the capital of Brazil?” and show the grounded refusal.
4. Optionally upload a small text document and query it.

## 4. Quality Evidence (60 seconds)

Show the automated test result and `evaluation/results-retrieval.json`. Explain confidence gating, strict chunk sizing, source citations, and the refusal test.

## 5. Limitations and Lessons (30 seconds)

Local CPU generation can be slower than cloud inference, scanned PDFs need OCR, and chunking plus retrieval thresholds materially affect answer quality.
