"""
Download/cache the Foundry Local models required by the RAG assistant.

This script needs internet once. After it finishes, the app can run offline.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundry_local_sdk import Configuration, FoundryLocalManager

from src.config import (
    FOUNDRY_APP_NAME,
    FOUNDRY_DIR,
    FOUNDRY_MODEL_CACHE_DIR,
    FOUNDRY_CHAT_MODEL_ALIAS,
    FOUNDRY_EMBEDDING_MODEL_ALIAS,
)


def download_model(alias: str) -> None:
    catalog = FoundryLocalManager.instance.catalog
    model = catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Foundry model alias not found: {alias}")

    print(f"{alias}: id={model.id} cached={model.is_cached} size={model.info.file_size_mb} MB")
    if model.is_cached:
        print(f"{alias}: already cached at {model.get_path()}")
        return

    print(f"Downloading {alias}...")

    def progress(percent: float) -> None:
        print(f"{alias}: {percent:5.1f}%")

    model.download(progress)
    print(f"{alias}: cached at {model.get_path()}")


def main() -> None:
    config = Configuration(
        app_name=FOUNDRY_APP_NAME,
        app_data_dir=FOUNDRY_DIR,
        model_cache_dir=FOUNDRY_MODEL_CACHE_DIR,
    )
    FoundryLocalManager.initialize(config)

    download_model(FOUNDRY_EMBEDDING_MODEL_ALIAS)
    download_model(FOUNDRY_CHAT_MODEL_ALIAS)
    print("\nFoundry Local setup complete. You can now run offline.")


if __name__ == "__main__":
    main()
