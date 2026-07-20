"""
database.py - SQLite storage for document chunks and embeddings.
"""

import json
import os
import sqlite3
from datetime import datetime

from src.config import DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_tables() -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            source    TEXT NOT NULL,
            content   TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            title      TEXT NOT NULL,
            messages   TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def clear_database() -> None:
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.commit()
    conn.close()


def insert_chunk(source: str, content: str, embedding: list[float]) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO chunks (source, content, embedding) VALUES (?, ?, ?)",
        (source, content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()


def get_all_chunks() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT id, source, content, embedding FROM chunks").fetchall()
    conn.close()

    result = []
    for row in rows:
        result.append({
            "id": row["id"],
            "source": row["source"],
            "content": row["content"],
            "embedding": json.loads(row["embedding"]),
        })
    return result


def get_chunks_by_source(source: str, limit: int | None = None) -> list[dict]:
    conn = get_connection()
    if limit is None:
        rows = conn.execute(
            "SELECT id, source, content, embedding FROM chunks WHERE source = ? ORDER BY id",
            (source,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, source, content, embedding FROM chunks WHERE source = ? ORDER BY id LIMIT ?",
            (source, limit),
        ).fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "source": row["source"],
            "content": row["content"],
            "embedding": json.loads(row["embedding"]),
            "score": 1.0,
            "semantic_score": 1.0,
            "keyword_score": 1.0,
        }
        for row in rows
    ]


def count_chunks() -> int:
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return count


def get_unique_sources() -> dict[str, int]:
    conn = get_connection()
    rows = conn.execute("SELECT source, COUNT(*) as count FROM chunks GROUP BY source").fetchall()
    conn.close()
    return {row["source"]: row["count"] for row in rows}


def delete_document_chunks(source: str) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
    conn.commit()
    conn.close()


def create_chat(title: str = "New chat") -> int:
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO chats (title, messages, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (title, "[]", now, now),
    )
    conn.commit()
    chat_id = cursor.lastrowid
    conn.close()
    return int(chat_id)


def update_chat(chat_id: int, title: str, messages: list[dict]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    conn = get_connection()
    conn.execute(
        "UPDATE chats SET title = ?, messages = ?, updated_at = ? WHERE id = ?",
        (title, json.dumps(messages), now, chat_id),
    )
    conn.commit()
    conn.close()


def get_chat(chat_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, title, messages, created_at, updated_at FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row["id"],
        "title": row["title"],
        "messages": json.loads(row["messages"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_chats(limit: int = 30) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM chats
        ORDER BY updated_at DESC, id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]


def delete_chat(chat_id: int) -> None:
    conn = get_connection()
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
