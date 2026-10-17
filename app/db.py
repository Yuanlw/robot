import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime

from .config import settings

DB_PATH = settings.data_dir / "robot.db"


@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                face_encoding BLOB,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                text TEXT NOT NULL,
                due_at TEXT NOT NULL,
                repeat_rule TEXT NOT NULL DEFAULT 'once',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                FOREIGN KEY(person_id) REFERENCES people(id)
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def record_event(event_type: str, payload: dict) -> None:
    with connection() as conn:
        conn.execute(
            "INSERT INTO events(event_type,payload,created_at) VALUES(?,?,?)",
            (event_type, json.dumps(payload, ensure_ascii=False), now_iso()),
        )


def list_events(limit: int = 30) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT id,event_type,payload,created_at FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"id": row["id"], "type": row["event_type"], "payload": json.loads(row["payload"]), "created_at": row["created_at"]}
        for row in rows
    ]
