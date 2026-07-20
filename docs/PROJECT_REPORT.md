# Project Report: Local RAG AI Assistant

## Purpose

This project is an offline question-answering assistant for private local documents. It uses Microsoft Foundry Local for both embeddings and answer generation, SQLite for local persistence, and Retrieval-Augmented Generation (RAG) to ground answers in retrieved document excerpts.

## Architecture

1. The loader extracts text from TXT, Markdown, PDF, and DOCX files.
2. The ingestion pipeline creates sentence-aware chunks with a strict 600-character ceiling.
3. Foundry Local's `qwen3-embedding-0.6b` model converts chunks to embeddings.
4. SQLite stores source names, chunk text, and serialized vectors.
5. Retrieval combines cosine similarity with keyword overlap and applies a confidence gate.
6. The `phi-3.5-mini` model answers using only the accepted context.
7. CLI and Streamlit interfaces display the answer, source names, and retrieved context.

## Grounding and Safety Controls

- Questions with no sufficiently relevant chunk are refused before the chat model runs.
- The prompt prohibits outside knowledge and requires an exact refusal when context is insufficient.
- A post-generation guard prevents the model from appending an outside fact after a refusal.
- Uploaded filenames are reduced to their basename before local storage.
- Documents and local model/database assets are excluded from version control by default.

## Testing and Evaluation

Automated tests cover vector similarity, database persistence, document loading, strict chunk sizing, confidence gating, refusal behavior, and a Foundry-backed integration path. The evaluation dataset includes answerable and deliberately unanswerable questions. Run:

```powershell
python -m pytest tests\ -v
python scripts/evaluate.py --mode retrieval --output evaluation/results-retrieval.json
python scripts/evaluate.py --mode full --output evaluation/results-full.json
```

Retrieval evaluation is fast. Full evaluation invokes the local LLM for each answerable case and can take several minutes on CPU-only hardware.

## Known Limitations

- Foundry generation speed depends strongly on the available CPU, GPU, or NPU.
- `pypdf` cannot extract useful text from image-only scans; those documents need OCR before ingestion.
- Brute-force cosine search is appropriate for this small teaching dataset, not a very large document collection.
- Confidence thresholds are calibrated for the included embedding model and should be re-evaluated if the model or document domain changes.

## Conclusion

The project meets the planned core deliverables: local inference, document ingestion, embeddings, SQLite storage, semantic retrieval, grounded generation, source display, offline operation, user interfaces, automated tests, and repeatable evaluation.
