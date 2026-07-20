"""
llm.py - Sends prompts to Microsoft Foundry Local.
"""

import threading
import time

from src.config import SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS
from src.foundry_runtime import get_chat_client

CHAT_TIMEOUT_SECONDS = 150


def _build_messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Use the following document excerpts to answer the question.\n\n"
                f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
                f"Question: {question}"
            ),
        },
    ]


def generate_answer(question: str, context: str) -> str:
    """Ask the Foundry Local chat model to answer using the provided context."""
    last_error = None
    for attempt in range(3):
        try:
            client = get_chat_client()
            client.settings.max_tokens = MAX_TOKENS
            client.settings.temperature = TEMPERATURE
            client.settings.top_p = 0.9
            client.settings.top_k = 40
            response = _complete_chat_with_timeout(client, _build_messages(question, context))
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            if "timed out" in error_text:
                break
            if "cancel" not in error_text or attempt == 2:
                break
            get_chat_client.cache_clear()
            time.sleep(1.0)

    raise RuntimeError(
        "The local chat model was busy right after indexing. Please ask again in a moment."
    ) from last_error


def _complete_chat_with_timeout(client, messages: list[dict]):
    result = {}

    def run_completion() -> None:
        try:
            result["response"] = client.complete_chat(messages)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=run_completion, daemon=True)
    thread.start()
    thread.join(CHAT_TIMEOUT_SECONDS)

    if thread.is_alive():
        raise TimeoutError(f"Local chat generation timed out after {CHAT_TIMEOUT_SECONDS} seconds.")

    if "error" in result:
        raise result["error"]

    return result["response"]
