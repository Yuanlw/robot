"""Explicit household speaker selection, not an authentication mechanism."""
import uuid
from .db import connection


def init_identity_db():
    with connection() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS speaker_bindings (
          browser_id TEXT PRIMARY KEY, person_id INTEGER, guest_key TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL DEFAULT 'guest')''')


def people():
    with connection() as c:
        return [dict(r) for r in c.execute('SELECT id,name FROM people ORDER BY id')]


def person(person_id):
    with connection() as c:
        row = c.execute('SELECT id,name FROM people WHERE id=?', (person_id,)).fetchone()
        return dict(row) if row else None


def current(browser_id):
    with connection() as c:
        inserted = c.execute('INSERT OR IGNORE INTO speaker_bindings(browser_id,guest_key) VALUES (?,?)',
                  (browser_id, 'guest:' + uuid.uuid4().hex))
        row = dict(c.execute('SELECT * FROM speaker_bindings WHERE browser_id=?', (browser_id,)).fetchone())
        # Adopt a pre-identity guest session only; never attach anonymous history to a member.
        if inserted.rowcount and not browser_id.startswith(('person:', 'guest:')):
            c.execute('UPDATE agent_sessions SET browser_id=? WHERE browser_id=?', (row['guest_key'], browser_id))
        member = c.execute('SELECT id,name FROM people WHERE id=?', (row['person_id'],)).fetchone()
    return {'person': dict(member) if member else None, 'revision': row['revision'],
            'source': row['source'], 'scope_key': 'person:' + str(member['id']) if member else row['guest_key']}


def public(browser_id):
    return {k: v for k, v in current(browser_id).items() if k != 'scope_key'}


def check(browser_id, revision):
    state = current(browser_id)
    if revision is not None and state['revision'] != revision:
        raise ValueError('当前说话人已改变，请刷新身份后再操作')
    return state


def select(browser_id, person_id, source='manual', revision=None):
    state = check(browser_id, revision)
    if person_id is not None and not person(person_id):
        raise ValueError('家庭成员不存在，请重新选择')
    current_id = state['person']['id'] if state['person'] else None
    # Re-recognizing the same member resumes the same conversation.
    if person_id is not None and person_id == current_id:
        return public(browser_id)
    with connection() as c:
        c.execute('UPDATE speaker_bindings SET person_id=?,guest_key=?,revision=revision+1,source=? WHERE browser_id=?',
                  (person_id, 'guest:' + uuid.uuid4().hex, source if person_id else 'guest', browser_id))
    return public(browser_id)


def visible_reminders(person_id):
    from .reminders import list_active
    return [r for r in list_active() if r['person_id'] is None or (person_id is not None and r['person_id'] == person_id)]


def require_reminder(reminder_id, person_id):
    from .reminders import get
    row = get(reminder_id)
    if not row or (row['person_id'] is not None and row['person_id'] != person_id):
        raise ValueError('当前成员无法操作这条提醒，请切换到对应成员')
    return row


def delete_person(person_id):
    """删除成员及本机人脸特征。返回该成员会话的 harness_id（可能为 None），供调用方回收常驻进程。

    安全语义：
    - 有未送达的个人提醒时拒绝，避免提醒悄悄失去归属；
    - 待确认草稿一并作废；历史事件文本与 data/harness 会话文件保留（与"新对话"一致，不物理删除）；
    - 正以该成员身份聊天的浏览器全部退回访客并递增 revision，页面周期身份同步会自动换屏。
    """
    person_id = int(person_id)
    with connection() as c:
        row = c.execute('SELECT id,name FROM people WHERE id=?', (person_id,)).fetchone()
        if not row:
            raise LookupError('成员不存在或已删除。')
        pending = c.execute("SELECT COUNT(*) FROM reminders WHERE person_id=? AND status='active'", (person_id,)).fetchone()[0]
        if pending:
            raise ValueError(f"{row['name']}还有 {pending} 条未完成的个人提醒，请先完成或取消，再删除成员。")
        scope = 'person:' + str(person_id)
        session = c.execute('SELECT harness_id FROM agent_sessions WHERE browser_id=?', (scope,)).fetchone()
        harness_id = session['harness_id'] if session else None
        if harness_id:
            c.execute("UPDATE agent_drafts SET status='cancelled' WHERE session_id=? AND status='draft'", (harness_id,))
            c.execute('DELETE FROM agent_clocks WHERE session_id=?', (harness_id,))
            c.execute('DELETE FROM agent_sessions WHERE browser_id=?', (scope,))
        c.execute("UPDATE agent_drafts SET status='cancelled' WHERE person_id=? AND status='draft'", (person_id,))
        c.execute('DELETE FROM people WHERE id=?', (person_id,))
        for binding in c.execute('SELECT browser_id FROM speaker_bindings WHERE person_id=?', (person_id,)).fetchall():
            # 每个浏览器独立 guest_key，避免被删成员的不同浏览器共享同一个访客会话
            c.execute("UPDATE speaker_bindings SET person_id=NULL,guest_key=?,revision=revision+1,source='guest' WHERE browser_id=?",
                      ('guest:' + uuid.uuid4().hex, binding['browser_id']))
        return harness_id
