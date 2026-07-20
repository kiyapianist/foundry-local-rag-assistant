"""
foundry_runtime.py - Foundry Local initialization and model clients.
"""

from __future__ import annotations

import os
from functools import lru_cache

from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.exception import FoundryLocalException

from src.config import (
    FOUNDRY_APP_NAME,
    FOUNDRY_DIR,
    FOUNDRY_MODEL_CACHE_DIR,
    FOUNDRY_CHAT_MODEL_ALIAS,
    FOUNDRY_EMBEDDING_MODEL_ALIAS,
    OFFLINE_MODE,
)


class FoundryRuntimeError(RuntimeError):
    """Raised when Foundry Local is not ready for offline inference."""


def configure_offline_environment() -> None:
    """Prevent runtime network calls from helper libraries."""
    if OFFLINE_MODE:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@lru_cache(maxsize=1)
def get_manager() -> FoundryLocalManager:
    configure_offline_environment()

    if FoundryLocalManager.instance is None:
        config = Configuration(
            app_name=FOUNDRY_APP_NAME,
            app_data_dir=FOUNDRY_DIR,
            model_cache_dir=FOUNDRY_MODEL_CACHE_DIR,
        )
        FoundryLocalManager.initialize(config)

    return FoundryLocalManager.instance


def _get_cached_model(alias: str):
    try:
        model = get_manager().catalog.get_model(alias)
    except Exception as exc:
        raise FoundryRuntimeError(
            "Foundry Local could not read the model catalog. Run "
            "`python scripts/setup_foundry.py` once while online, then start "
            "the app again in offline mode."
        ) from exc

    if model is None:
        raise FoundryRuntimeError(f"Foundry model alias not found: {alias}")

    try:
        cached = model.is_cached
    except Exception as exc:
        raise FoundryRuntimeError(
            f"Foundry model '{alias}' could not be checked in the local cache."
        ) from exc

    if not cached:
        raise FoundryRuntimeError(
            f"Foundry model '{alias}' is not cached. Run: python scripts/setup_foundry.py"
        )

    return model


@lru_cache(maxsize=1)
def get_embedding_client():
    model = _get_cached_model(FOUNDRY_EMBEDDING_MODEL_ALIAS)
    model.load()
    return model.get_embedding_client()


@lru_cache(maxsize=1)
def get_chat_client():
    model = _get_cached_model(FOUNDRY_CHAT_MODEL_ALIAS)
    model.load()
    return model.get_chat_client()


def get_runtime_status() -> dict:
    status = {
        "offline_mode": OFFLINE_MODE,
        "chat_alias": FOUNDRY_CHAT_MODEL_ALIAS,
        "embedding_alias": FOUNDRY_EMBEDDING_MODEL_ALIAS,
        "foundry_ready": False,
        "chat_cached": False,
        "embedding_cached": False,
        "error": None,
    }
    try:
        chat_model = _get_cached_model(FOUNDRY_CHAT_MODEL_ALIAS)
        embedding_model = _get_cached_model(FOUNDRY_EMBEDDING_MODEL_ALIAS)
        status["chat_cached"] = chat_model.is_cached
        status["embedding_cached"] = embedding_model.is_cached
        status["foundry_ready"] = status["chat_cached"] and status["embedding_cached"]
    except Exception as exc:
        status["error"] = str(exc)
    return status


def foundry_exception_message(exc: Exception) -> str:
    if isinstance(exc, FoundryLocalException):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"
