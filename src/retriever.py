"""
retriever.py - Finds the most relevant document chunks for a user query.
"""

import re

import numpy as np

from src.config import (
    MIN_KEYWORD_SCORE,
    MIN_RETRIEVAL_SCORE,
    MIN_SEMANTIC_SCORE,
    TOP_K,
)
from src.database import get_all_chunks
from src.embedder import embed_text


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Measure how similar two vectors are."""
    a = np.array(vec_a)
    b = np.array(vec_b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def _passes_confidence_gate(chunk: dict) -> bool:
    """Require credible semantic evidence or a semantic/keyword combination."""
    return (
        chunk["semantic_score"] >= MIN_SEMANTIC_SCORE
        or (
            chunk["score"] >= MIN_RETRIEVAL_SCORE
            and chunk["keyword_score"] >= MIN_KEYWORD_SCORE
        )
    )


def find_relevant_chunks(query: str, top_k: int = TOP_K, allowed_sources: list[str] | None = None) -> list[dict]:
    """Return the top matching chunks using semantic and keyword signals."""
    query_vector = embed_text(query)
    all_chunks = get_all_chunks()

    if not all_chunks:
        return []

    if allowed_sources is not None:
        allowed_set = set(allowed_sources)
        all_chunks = [c for c in all_chunks if c["source"] in allowed_set]

    if not all_chunks:
        return []

    stop_words = {
        "the", "a", "an", "in", "on", "at", "for", "to", "of", "and", "or",
        "is", "are", "what", "tell", "me", "show", "get", "find", "pdf",
        "document",
    }
    query_words = re.findall(r"\w+", query.lower())
    query_words = [w for w in query_words if w not in stop_words and len(w) > 1]

    scored = []
    for chunk in all_chunks:
        semantic_score = cosine_similarity(query_vector, chunk["embedding"])

        content_lower = chunk["content"].lower()
        if query_words:
            matches = sum(1 for word in query_words if word in content_lower)
            keyword_score = matches / len(query_words)
        else:
            keyword_score = 0.0

        combined_score = 0.7 * semantic_score + 0.3 * keyword_score
        scored.append({
            "id": chunk["id"],
            "source": chunk["source"],
            "content": chunk["content"],
            "score": combined_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_score,
        })

    scored = [chunk for chunk in scored if _passes_confidence_gate(chunk)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored[:top_k]
    top_chunks.sort(key=lambda x: x["id"])
    return top_chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into context for the LLM prompt."""
    if not chunks:
        return "No relevant documents found."

    parts = []
    for chunk in chunks:
        parts.append(f"[Source: {chunk['source']}]\n{chunk['content']}")

    return "\n\n".join(parts)
