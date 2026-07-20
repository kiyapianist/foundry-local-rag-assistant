"""
embedder.py - Generates text embeddings with Microsoft Foundry Local.
"""

from src.foundry_runtime import get_embedding_client


def embed_text(text: str) -> list[float]:
    """Convert a single string into a vector."""
    response = get_embedding_client().generate_embedding(text)
    return response.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into vectors."""
    if not texts:
        return []
    response = get_embedding_client().generate_embeddings(texts)
    return [item.embedding for item in response.data]
