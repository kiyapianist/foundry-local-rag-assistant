"""
scripts/ingest.py - Load documents into the SQLite knowledge base.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DOCUMENTS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.database import create_tables, clear_database, insert_chunk, count_chunks
from src.embedder import embed_batch
from src.loader import load_single_document


import re

MIN_DOCUMENT_TEXT_LENGTH = 40


def _split_oversized_text(text: str, max_size: int) -> list[str]:
    """Split one oversized sentence/paragraph at word boundaries."""
    pieces = []
    remaining = text.strip()
    while len(remaining) > max_size:
        cut = remaining.rfind(" ", 0, max_size + 1)
        if cut < max_size // 2:
            cut = max_size
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def _overlap_tail(text: str, overlap: int) -> str:
    """Return at most ``overlap`` trailing characters on a word boundary."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and " " in tail:
        tail = tail.split(" ", 1)[1]
    return tail.strip()

def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into sentence-aware chunks with a strict size ceiling."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split by sentence boundaries (periods, question marks, exclamation marks followed by whitespace)
    # Avoid splitting on abbreviations
    sentence_endings = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s")
    sentences = sentence_endings.split(text)

    units = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            units.extend(_split_oversized_text(sentence, chunk_size))

    chunks = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        tail = _overlap_tail(current, overlap)
        available_overlap = max(0, chunk_size - len(unit) - 1)
        if len(tail) > available_overlap:
            tail = _overlap_tail(tail, available_overlap)
        current = f"{tail} {unit}".strip()

    if current:
        chunks.append(current)

    if any(len(chunk) > chunk_size for chunk in chunks):
        raise AssertionError("chunking produced a chunk larger than chunk_size")

    return chunks


def load_documents(directory: str) -> list[dict]:
    """Read all supported files from a directory."""
    docs = []
    if not os.path.exists(directory):
        print(f"[ERROR] Documents folder not found: {directory}")
        return docs

    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith((".txt", ".pdf", ".docx", ".md")):
            filepath = os.path.join(directory, filename)
            if os.path.getsize(filepath) == 0:
                print(f"  SKIP Empty file: {filename}")
                continue
            try:
                content = load_single_document(filepath, filename)
                if len(content.strip()) >= MIN_DOCUMENT_TEXT_LENGTH:
                    docs.append({"filename": filename, "content": content})
                    print(f"  OK Loaded: {filename} ({len(content)} chars)")
                else:
                    print(
                        f"  SKIP Too little extractable text: {filename} "
                        f"({len(content.strip())} chars; scanned documents may need OCR)"
                    )
            except Exception as e:
                print(f"  FAIL Failed to load {filename}: {e}")

    return docs


def ingest(force: bool = False):
    print("\nStarting document ingestion...\n")

    create_tables()

    current = count_chunks()
    if current > 0 and not force:
        answer = input(f"  Database has {current} chunks. Clear and re-ingest? [y/N]: ").strip().lower()
        if answer == "y":
            clear_database()
        else:
            print("  Keeping existing data. Exiting.")
            return
    elif current > 0 and force:
        clear_database()

    print("Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print(f"\n[ERROR] No .txt, .pdf, .docx, or .md files found in {DOCUMENTS_DIR}")
        print("  Add supported files and run again.")
        return

    print(f"\nSplitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []

    for doc in documents:
        chunks = split_into_chunks(doc["content"], CHUNK_SIZE, CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append((doc["filename"], chunk))
        print(f"  {doc['filename']}: {len(chunks)} chunks")

    print(f"\n  Total chunks to embed: {len(all_chunks)}")
    print("\nGenerating Foundry Local embeddings (this may take a minute)...")

    texts = [c[1] for c in all_chunks]
    filenames = [c[0] for c in all_chunks]

    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_batch(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embedded {min(i + batch_size, len(texts))} / {len(texts)}")

    print("\nSaving to database...")
    for filename, text, embedding in zip(filenames, texts, all_embeddings):
        insert_chunk(source=filename, content=text, embedding=embedding)

    final_count = count_chunks()
    print(f"\nDone! {final_count} chunks stored in database.")
    print("   Database: data/knowledge.db")
    print("\n   Now run: python main.py  (or)  streamlit run app.py\n")


if __name__ == "__main__":
    ingest(force="--yes" in sys.argv or "-y" in sys.argv)
