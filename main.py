"""
main.py - Command-line interface for the RAG assistant.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import count_chunks
from src.foundry_runtime import get_runtime_status
from src.pipeline import ask


def print_separator() -> None:
    print("\n" + "-" * 60 + "\n")


def main() -> None:
    print("\n" + "=" * 60)
    print("   Local RAG Assistant - Foundry Local")
    print("=" * 60)

    runtime = get_runtime_status()
    if not runtime["foundry_ready"]:
        print("\nFoundry Local models are not ready.")
        if runtime["error"]:
            print(runtime["error"])
        print("Run: python scripts/setup_foundry.py")
        return

    try:
        n_chunks = count_chunks()
        if n_chunks == 0:
            print("\nKnowledge base is empty.")
            print("Run: python scripts/ingest.py")
            return
        print(f"\nKnowledge base ready ({n_chunks} chunks loaded)")
    except Exception as exc:
        print(f"\nDatabase error: {exc}")
        print("Run: python scripts/ingest.py")
        return

    print("\nType your question below. Type 'quit' to exit.")
    print_separator()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("\nGoodbye.")
            break

        print("\nThinking...")
        result = ask(question)
        print(f"\nAssistant:\n{result['answer']}")

        if result["sources"]:
            print(f"\nSources: {', '.join(result['sources'])}")

        print_separator()


if __name__ == "__main__":
    main()
