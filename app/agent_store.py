"""Robot-owned sessions and reminder drafts. Harness never writes active reminders directly."""
import json
import uuid
from datetime import datetime, timedelta

from .db import connection, now_iso
from . import identity


def init_agent_db():
    with connection() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS agent_sessions (
          browser_id TEXT PRIMARY KEY, harness_id TEXT NOT NULL, started INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS agent_clocks (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS agent_drafts (
          id TEXT PRIMARY KEY, session_id TEXT NOT NULL, text TEXT NOT NULL, due_at TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'draft',
          target_id INTEGER, reminder_id INTEGER, created_at TEXT NOT NULL);
        """)

        columns = {r['name'] for r in c.execute('PRAGMA table_info(agent_drafts)')}
        if 'person_id' not in columns:
            c.execute('ALTER TABLE agent_drafts ADD COLUMN person_id INTEGER')
    identity.init_identity_db()


def session(browser_id):
    state = identity.current(browser_id)
    browser_id = state['scope_key']
    with connection() as c:
        c.execute("INSERT OR IGNORE INTO agent_sessions(browser_id,harness_id) VALUES (?,?)",
                  (browser_id, 'robot-' + uuid.uuid4().hex))
        return dict(c.execute("SELECT * FROM agent_sessions WHERE browser_id=?", (browser_id,)).fetchone())


def mark_started(browser_id):
    browser_id = identity.current(browser_id)['scope_key']
    with connection() as c:
        c.execute("UPDATE agent_sessions SET started=1 WHERE browser_id=?", (browser_id,))


def clear(browser_id):
    browser_id = identity.current(browser_id)['scope_key']
    with connection() as c:
        row = c.execute("SELECT harness_id FROM agent_sessions WHERE browser_id=?", (browser_id,)).fetchone()
        if row:
            c.execute("UPDATE agent_drafts SET status='cancelled' WHERE session_id=? AND status='draft'", (row[0],))
            c.execute("DELETE FROM agent_clocks WHERE session_id=?", (row[0],))
        c.execute("DELETE FROM agent_sessions WHERE browser_id=?", (browser_id,))


def drafts(sid):
    with connection() as c:
        return [dict(r) for r in c.execute("SELECT * FROM agent_drafts WHERE session_id=? AND status='draft' ORDER BY created_at", (sid,))]


def future(value):
    due = datetime.fromisoformat(value)
    if due.tzinfo is None or due <= datetime.now().astimezone():
        raise ValueError("提醒时间必须带时区且晚于现在，请重新确认时间")
    return due


def prepare(sid, args, received_at, person_id=None):
    with connection() as c:
        c.execute("BEGIN IMMEDIATE")
        old = None
        if args.get('draft_id'):
            old = c.execute("SELECT * FROM agent_drafts WHERE id=? AND session_id=? AND status='draft'",
                            (args['draft_id'], sid)).fetchone()
            if not old:
                raise ValueError("草稿不存在或已经处理，请重新查询提醒")
        target_id = old['target_id'] if old else args.get('reminder_id')
        target = None
        if target_id is not None:
            target = c.execute("SELECT * FROM reminders WHERE id=? AND status='active'", (target_id,)).fetchone()
            if not target:
                raise ValueError("原提醒已送达或取消，不能修改")
        if target and target['person_id'] is not None and target['person_id'] != person_id:
            raise ValueError('当前成员无法修改这条提醒')
        base = old or target
        scope = args.get('scope')
        if scope not in (None, 'personal', 'household'):
            raise ValueError('提醒归属不正确')
        if scope == 'personal' and person_id is None:
            raise ValueError('请先识别或选择家庭成员，再设置个人提醒')
        owner = base['person_id'] if base else person_id
        if scope == 'household':
            owner = None
        elif scope == 'personal':
            owner = person_id
        if target and owner != target['person_id']:
            raise ValueError('修改已有提醒不能改变归属，请另建提醒')
        text = args.get('text', base['text'] if base else '').strip()
        if text in ('', '提醒', '提醒我', '待补充', '待定', '未指定') or len(text) > 300:
            raise ValueError("请先询问具体提醒事项，不能创建占位草稿；内容最多300字")
        if 'delay_seconds' in args:
            delay = args['delay_seconds']
            if 'due_at' in args or type(delay) is not int or not 0 < delay <= 31536000:
                raise ValueError("相对时间须为正整数秒，不可同时传绝对时间")
            clock = c.execute("SELECT at FROM agent_clocks WHERE id=? AND session_id=?",
                              (args.get('reference_id'), sid)).fetchone()
            if not clock:
                raise ValueError("先调用 get_current_time 取得时间基准")
            # Updates with a new duration start at this input; content-only edits omit duration.
            anchor = received_at if base else clock['at']
            due = (datetime.fromisoformat(anchor) + timedelta(seconds=delay)).isoformat()
        elif 'due_at' in args:
            due = args['due_at']
        elif base:
            due = base['due_at']
        else:
            raise ValueError("缺少时间，请询问用户")
        future(due)
        if old:
            rid = old['id']
            c.execute("UPDATE agent_drafts SET text=?,due_at=?,person_id=?,revision=revision+1 WHERE id=?", (text, due, owner, rid))
        else:
            # One pending draft per conversation makes voice confirmation unambiguous.
            c.execute("UPDATE agent_drafts SET status='superseded' WHERE session_id=? AND status='draft'", (sid,))
            rid = uuid.uuid4().hex
            c.execute("INSERT INTO agent_drafts(id,session_id,text,due_at,target_id,created_at,person_id) VALUES (?,?,?,?,?,?,?)",
                      (rid, sid, text, due, target_id, now_iso(), owner))
        return dict(c.execute("SELECT * FROM agent_drafts WHERE id=?", (rid,)).fetchone())


def confirm(sid, draft_id, revision):
    """An atomic and idempotent commit shared by button and authorized voice confirmation."""
    with connection() as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute("SELECT * FROM agent_drafts WHERE id=? AND session_id=?", (draft_id, sid)).fetchone()
        if not row or row['revision'] != revision:
            raise ValueError("草稿已修改或不存在，请确认最新内容")
        if row['status'] == 'confirmed':
            return dict(c.execute("SELECT * FROM reminders WHERE id=?", (row['reminder_id'],)).fetchone())
        if row['status'] != 'draft':
            raise ValueError("草稿已取消或被替换")
        future(row['due_at'])
        if row['target_id'] is not None:
            rid = row['target_id']
            if c.execute("UPDATE reminders SET text=?,due_at=? WHERE id=? AND status='active'",
                         (row['text'], row['due_at'], rid)).rowcount != 1:
                raise ValueError("原提醒已送达或取消，请重新查询")
        else:
            rid = c.execute("INSERT INTO reminders(text,due_at,created_at,person_id) VALUES (?,?,?,?)",
                            (row['text'], row['due_at'], now_iso(), row['person_id'])).lastrowid
        c.execute("UPDATE agent_drafts SET status='confirmed',reminder_id=? WHERE id=?", (rid, draft_id))
        c.execute("INSERT INTO events(event_type,payload,created_at) VALUES (?,?,?)",
                  ('reminder_saved', json.dumps({'reminder_id': rid, 'text': row['text']}, ensure_ascii=False), now_iso()))
        return dict(c.execute("SELECT * FROM reminders WHERE id=?", (rid,)).fetchone())


def cancel_draft(sid, draft_id):
    with connection() as c:
        if c.execute("UPDATE agent_drafts SET status='cancelled' WHERE id=? AND session_id=? AND status='draft'", (draft_id, sid)).rowcount != 1:
            raise ValueError("未找到待确认草稿")
    return {'cancelled': True, 'draft_id': draft_id}
