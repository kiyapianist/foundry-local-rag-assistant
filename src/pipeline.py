"""
pipeline.py - The RAG pipeline.

Flow:
  User question -> retrieve chunks -> build prompt -> Foundry Local -> answer
"""

import re

from src.database import count_chunks, get_chunks_by_source
from src.llm import generate_answer
from src.retriever import find_relevant_chunks, format_context

NO_INFORMATION_ANSWER = "I don't have that information in my documents."


def _is_summary_question(question: str) -> bool:
    normalized = question.lower()
    return any(word in normalized for word in ("summar", "summary", "overview", "main idea", "main point"))


def _clean_sentence(text: str) -> str:
    sentence = re.sub(r"\s+", " ", text).strip()
    return sentence[:320].rstrip(" ,;:-") + ("..." if len(sentence) > 320 else "")


def _merge_chunk_text(chunks: list[dict]) -> str:
    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.get("id", 0))
    merged = ""
    for chunk in ordered_chunks:
        text = chunk["content"].strip()
        if not text:
            continue

        if not merged:
            merged = text
            continue

        overlap = 0
        max_overlap = min(len(merged), len(text), 160)
        for size in range(max_overlap, 20, -1):
            if merged[-size:] == text[:size]:
                overlap = size
                break
        merged += " " + text[overlap:].lstrip()

    return re.sub(r"\s+", " ", merged).strip()


def _split_document_points(text: str, limit: int = 14) -> list[str]:
    points = []
    seen = set()
    parts = re.split(
        r"(?=\b(?:LEVEL|Level)\s+\d+\b)|(?=\b\d+\s*[️⃣.)-])|(?<=[.!?])\s+",
        text,
    )
    for part in parts:
        cleaned = _clean_sentence(part)
        key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
        if len(cleaned) < 22 or key in seen:
            continue
        points.append(cleaned)
        seen.add(key)
        if len(points) >= limit:
            return points
    return points


def _relevant_points(question: str, text: str, limit: int = 8) -> list[str]:
    query_words = _query_words(question)
    points = _split_document_points(text, limit=50)
    if not query_words:
        return points[:limit]

    ranked = []
    for point in points:
        point_lower = point.lower()
        score = sum(point_lower.count(word) for word in query_words)
        ranked.append((score, point))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [point for score, point in ranked if score > 0][:limit]
    return selected or points[:limit]


def _query_words(question: str) -> list[str]:
    stop_words = {
        "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "or",
        "is", "are", "what", "tell", "me", "show", "get", "find", "pdf",
        "document", "documnet", "doc", "give", "please", "about",
    }
    words = re.findall(r"\w+", question.lower())
    return [word for word in words if len(word) > 2 and word not in stop_words]


def _rank_source_chunks(question: str, chunks: list[dict], limit: int = 12) -> list[dict]:
    if _is_summary_question(question):
        return chunks[:20]

    query_words = _query_words(question)
    if not query_words:
        return chunks[:limit]

    ranked = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        score = sum(content_lower.count(word) for word in query_words)
        ranked.append({**chunk, "score": float(score)})

    ranked.sort(key=lambda chunk: (chunk["score"], -chunk["id"]), reverse=True)
    top_chunks = [chunk for chunk in ranked if chunk["score"] > 0][:limit]
    return top_chunks or chunks[:limit]


def _summary_context_chunks(sources: list[str], max_chunks: int = 10) -> list[dict]:
    """Sample a document from beginning to end for an explicit summary request."""
    selected = []
    for source in sources[:3]:
        source_chunks = get_chunks_by_source(source)
        if not source_chunks:
            continue
        remaining = max_chunks - len(selected)
        if remaining <= 0:
            break
        take = min(remaining, len(source_chunks))
        if take == 1:
            indices = [0]
        else:
            indices = [round(i * (len(source_chunks) - 1) / (take - 1)) for i in range(take)]
        selected.extend(source_chunks[index] for index in dict.fromkeys(indices))
    return selected


def fallback_answer(question: str, chunks: list[dict], error: Exception | None = None) -> str:
    if not chunks:
        return NO_INFORMATION_ANSWER

    sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))
    source_text = ", ".join(sources)
    document_text = _merge_chunk_text(chunks)

    if _is_summary_question(question):
        snippets = _split_document_points(document_text, limit=14)

        bullets = "\n".join(f"- {snippet}" for snippet in snippets)
        return (
            f"Based on {source_text}, here is the summary:\n\n"
            f"{bullets}"
        )

    relevant = _relevant_points(question, document_text, limit=8)
    best_text = "\n".join(f"- {part}" for part in relevant)
    return (
        f"Based on {source_text}, the most relevant information I found is:\n\n"
        f"{best_text}"
    )


def _sanitize_grounded_answer(answer: str) -> str:
    """Prevent a model from appending outside knowledge after its refusal."""
    normalized = answer.strip()
    refusal_markers = (
        "i don't have that information in my documents",
        "i do not have that information in my documents",
        "not mentioned in the provided context",
        "not present in the provided context",
        "cannot be determined from the context",
    )
    if any(marker in normalized.lower() for marker in refusal_markers):
        return NO_INFORMATION_ANSWER
    return normalized


def ask(question: str, preferred_sources: list[str] | None = None) -> dict:
    """Run the full RAG pipeline for one user question."""
    if not question or not question.strip():
        return {
            "question": question,
            "answer": "Please enter a question.",
            "sources": [],
            "chunks": [],
        }

    if count_chunks() == 0:
        return {
            "question": question,
            "answer": "Knowledge base is empty. Run `python scripts/ingest.py` first.",
            "sources": [],
            "chunks": [],
        }

    # Explicit document summaries need coverage across the source, not only top matches.
    if preferred_sources and _is_summary_question(question):
        chunks = _summary_context_chunks(preferred_sources)
    else:
        chunks = find_relevant_chunks(
            question,
            allowed_sources=preferred_sources,
        )

    if not chunks:
        return {
            "question": question,
            "answer": NO_INFORMATION_ANSWER,
            "sources": [],
            "chunks": [],
        }

    context = format_context(chunks)
    try:
        answer = _sanitize_grounded_answer(generate_answer(question, context))
    except Exception as exc:
        answer = fallback_answer(question, chunks, exc)

    sources = list(dict.fromkeys(c["source"] for c in chunks))

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
    }
