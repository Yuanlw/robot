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
            CREATE TABLE IF NOT EXISTS conversation_memories (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL DEFAULT '',
                messages TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_key TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_turns_scope ON conversation_turns(scope_key, id);
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


def get_conversation_memory(session_id: str) -> dict:
    init_db()
    with connection() as conn:
        row = conn.execute(
            "SELECT summary,messages FROM conversation_memories WHERE session_id=?", (session_id,)
        ).fetchone()
    return {"summary": row["summary"], "messages": json.loads(row["messages"])} if row else {"summary": "", "messages": []}


def save_conversation_memory(session_id: str, summary: str, messages: list[dict]) -> None:
    init_db()
    with connection() as conn:
        conn.execute(
            "INSERT INTO conversation_memories(session_id,summary,messages,updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary,messages=excluded.messages,updated_at=excluded.updated_at",
            (session_id, summary, json.dumps(messages, ensure_ascii=False), now_iso()),
        )


def delete_conversation_memory(session_id: str) -> None:
    init_db()
    with connection() as conn:
        conn.execute("DELETE FROM conversation_memories WHERE session_id=?", (session_id,))


def record_turn(scope_key: str, role: str, text: str) -> None:
    """逐轮记录对话转录（按身份 scope_key 归属），供刷新页面后恢复可见聊天。

    与 conversation_memories（legacy 摘要记忆）分工：这里只存可见转录，
    不参与模型上下文；harness 后端的模型记忆仍在 dsh 会话里。
    """
    text = (text or "").strip()
    if not text:
        return
    with connection() as conn:
        conn.execute("INSERT INTO conversation_turns(scope_key,role,text,created_at) VALUES(?,?,?,?)",
                     (scope_key, role, text[:4000], now_iso()))


def list_turns(scope_key: str, limit: int = 40) -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT role,text,created_at FROM conversation_turns WHERE scope_key=? ORDER BY id DESC LIMIT ?",
            (scope_key, limit)).fetchall()
    return [{"role": r["role"], "text": r["text"], "created_at": r["created_at"]} for r in reversed(rows)]


def delete_turns(scope_key: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM conversation_turns WHERE scope_key=?", (scope_key,))
