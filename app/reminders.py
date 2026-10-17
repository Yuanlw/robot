from datetime import datetime

from .db import connection, now_iso


def create(text: str, due_at: datetime, person_id: int | None = None, repeat_rule: str = "once") -> dict:
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO reminders(person_id,text,due_at,repeat_rule,created_at) VALUES(?,?,?,?,?)",
            (person_id, text, due_at.isoformat(), repeat_rule, now_iso()),
        )
        reminder_id = cur.lastrowid
    return get(reminder_id)


def get(reminder_id: int) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT r.*, p.name AS person_name FROM reminders r LEFT JOIN people p ON p.id=r.person_id WHERE r.id=?",
            (reminder_id,),
        ).fetchone()
        return dict(row) if row else None


def list_active() -> list[dict]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT r.*, p.name AS person_name FROM reminders r LEFT JOIN people p ON p.id=r.person_id WHERE r.status='active' ORDER BY due_at"
        ).fetchall()
        return [dict(row) for row in rows]


def cancel(reminder_id: int) -> bool:
    with connection() as conn:
        cur = conn.execute("UPDATE reminders SET status='cancelled' WHERE id=? AND status='active'", (reminder_id,))
        return cur.rowcount > 0


def due_now(now: datetime | None = None) -> list[dict]:
    """Return due reminders and mark one-shot reminders as delivered."""
    current = (now or datetime.now().astimezone()).isoformat()
    with connection() as conn:
        rows = conn.execute(
            "SELECT r.*, p.name AS person_name FROM reminders r LEFT JOIN people p ON p.id=r.person_id "
            "WHERE r.status='active' AND r.due_at <= ? ORDER BY r.due_at",
            (current,),
        ).fetchall()
        due = [dict(row) for row in rows]
        for row in rows:
            if row["repeat_rule"] == "once":
                conn.execute("UPDATE reminders SET status='delivered' WHERE id=?", (row["id"],))
        return due
