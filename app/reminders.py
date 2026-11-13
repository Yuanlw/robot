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
            "SELECT r.*, p.name AS person_name FROM reminders r LEFT JOIN people p ON p.id=r.person_id WHERE r.status='active' ORDER BY julianday(r.due_at)"
        ).fetchall()
        return [dict(row) for row in rows]


def cancel(reminder_id: int) -> bool:
    with connection() as conn:
        cur = conn.execute("UPDATE reminders SET status='cancelled' WHERE id=? AND status='active'", (reminder_id,))
        return cur.rowcount > 0


def update(reminder_id: int, text: str, due_at: datetime) -> dict | None:
    with connection() as conn:
        row = conn.execute(
            "SELECT id FROM reminders WHERE id=? AND status='active'", (reminder_id,)
        ).fetchone()
        if not row:
            return None
        cur = conn.execute(
            "UPDATE reminders SET text=?,due_at=? WHERE id=? AND status='active'",
            (text, due_at.isoformat(), reminder_id),
        )
        if cur.rowcount == 0:
            return None
    return get(reminder_id)


def due_now(now: datetime | None = None, expiry_hours: int | None = 24) -> list[dict]:
    """Read due reminders; delivery is acknowledged only after successful playback.

    expiry_hours: 超过此小时数的过期提醒不再自动播放。None 则不限制过期时间。
    注意：用户明确设定的提醒到点必须触发（"早上7点吃药"就是7点响），
    安静时段只约束机器人自主生成的播报（学习/坐姿，见 study_loop）。
    陈旧补播风险由 expiry_hours 兜底（2026-11-13 修正误伤的安静时段）。
    """
    current_dt = now or datetime.now().astimezone()
    current = current_dt.isoformat()

    with connection() as conn:
        if expiry_hours is None:
            rows = conn.execute(
                "SELECT r.*, p.name AS person_name FROM reminders r LEFT JOIN people p ON p.id=r.person_id "
                "WHERE r.status='active' AND julianday(r.due_at) <= julianday(?) "
                "ORDER BY julianday(r.due_at)",
                (current,),
            ).fetchall()
        else:
            expiry_threshold = current_dt.timestamp() - expiry_hours * 3600
            expiry_iso = datetime.fromtimestamp(expiry_threshold, tz=current_dt.tzinfo).isoformat()
            rows = conn.execute(
                "SELECT r.*, p.name AS person_name FROM reminders r LEFT JOIN people p ON p.id=r.person_id "
                "WHERE r.status='active' AND julianday(r.due_at) <= julianday(?) "
                "AND julianday(r.due_at) >= julianday(?) "
                "ORDER BY julianday(r.due_at)",
                (current, expiry_iso),
            ).fetchall()
        due = [dict(row) for row in rows]
        return due


def mark_delivered(reminder_id: int) -> None:
    with connection() as conn:
        conn.execute("UPDATE reminders SET status='delivered' WHERE id=? AND status='active'", (reminder_id,))
