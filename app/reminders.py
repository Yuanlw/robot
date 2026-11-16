from datetime import datetime, timedelta

from .db import connection, now_iso

REPEAT_RULES = ("once", "daily", "weekly")
_REPEAT_DELTA = {"daily": timedelta(days=1), "weekly": timedelta(weeks=1)}


def create(text: str, due_at: datetime, person_id: int | None = None, repeat_rule: str = "once") -> dict:
    if repeat_rule not in REPEAT_RULES:
        raise ValueError("重复规则不正确")
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


def update(reminder_id: int, text: str, due_at: datetime, repeat_rule: str | None = None) -> dict | None:
    if repeat_rule is not None and repeat_rule not in REPEAT_RULES:
        raise ValueError("重复规则不正确")
    with connection() as conn:
        row = conn.execute(
            "SELECT id FROM reminders WHERE id=? AND status='active'", (reminder_id,)
        ).fetchone()
        if not row:
            return None
        if repeat_rule is None:
            cur = conn.execute(
                "UPDATE reminders SET text=?,due_at=? WHERE id=? AND status='active'",
                (text, due_at.isoformat(), reminder_id),
            )
        else:
            cur = conn.execute(
                "UPDATE reminders SET text=?,due_at=?,repeat_rule=? WHERE id=? AND status='active'",
                (text, due_at.isoformat(), repeat_rule, reminder_id),
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
    重复提醒（daily/weekly）豁免过期下限：送达后自行推进到下一次，
    静音/勿扰期间保持 active 重试，不会因超过 24h 被悄悄丢弃。
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
                "AND (r.repeat_rule != 'once' OR julianday(r.due_at) >= julianday(?)) "
                "ORDER BY julianday(r.due_at)",
                (current, expiry_iso),
            ).fetchall()
        due = [dict(row) for row in rows]
        return due


def mark_delivered(reminder_id: int, now: datetime | None = None) -> None:
    """一次性提醒标送达；重复提醒推进到下一次（保持 active）。

    推进锚定原到期钟点（每天同一时刻），并跳过错过的次数——
    静音/停机几天后恢复，只播当前这一次，不补播陈旧话术。
    """
    current_dt = now or datetime.now().astimezone()
    with connection() as conn:
        row = conn.execute("SELECT due_at,repeat_rule FROM reminders WHERE id=? AND status='active'",
                           (reminder_id,)).fetchone()
        if not row:
            return
        rule = row["repeat_rule"]
        if rule in _REPEAT_DELTA:
            due = datetime.fromisoformat(row["due_at"])
            delta = _REPEAT_DELTA[rule]
            nxt = due + delta
            while nxt <= current_dt:
                nxt += delta
            conn.execute("UPDATE reminders SET due_at=? WHERE id=? AND status='active'",
                         (nxt.isoformat(), reminder_id))
        else:
            conn.execute("UPDATE reminders SET status='delivered' WHERE id=? AND status='active'", (reminder_id,))
