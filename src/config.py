"""
config.py - All project settings in one place.
Change values here to customize the assistant.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "data", "documents")
DATABASE_PATH = os.path.join(BASE_DIR, "data", "knowledge.db")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
FOUNDRY_DIR = os.path.join(BASE_DIR, "data", "foundry")
FOUNDRY_MODEL_CACHE_DIR = os.path.join(FOUNDRY_DIR, "models")

os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FOUNDRY_MODEL_CACHE_DIR, exist_ok=True)

# Foundry Local model settings
FOUNDRY_APP_NAME = "rag-assistant"
FOUNDRY_CHAT_MODEL_ALIAS = "phi-3.5-mini"
FOUNDRY_EMBEDDING_MODEL_ALIAS = "qwen3-embedding-0.6b"

# Runtime defaults
OFFLINE_MODE = os.getenv("RAG_OFFLINE_MODE", "1").lower() not in {"0", "false", "no"}

# Chunking settings
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Retrieval settings
TOP_K = 4
MIN_RETRIEVAL_SCORE = 0.34
MIN_SEMANTIC_SCORE = 0.48
MIN_KEYWORD_SCORE = 0.20

# LLM generation settings
MAX_TOKENS = 256
TEMPERATURE = 0.20

# System prompt
SYSTEM_PROMPT = """You are an expert, helpful, and precise local AI knowledge assistant.
Your task is to answer the user's questions clearly, accurately, and naturally, using the provided document context.
- The context blocks are formatted in chronological order of the original document.
- If the user asks for structural/layout details, use the top or beginning of the provided context.
- Provide a helpful, complete answer based only on facts in the context.
- Be concise but natural.
  - Never use outside knowledge, even when you know the answer.
  - If the answer is not present or cannot be determined from the context, reply with exactly: "I don't have that information in my documents."
  - Do not add an explanation or an outside fact after that refusal.
  - Always mention the source document name where the answer was found."""
