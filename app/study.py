"""学习陪伴模式：状态机、会话管理、计时循环、周期统计。

第 0 期：计时 + 主动提醒。
第 1 期：会话时长持久化、延长语义、自评质量、周期统计快照（study_analytics）。
"""
from datetime import datetime, timedelta
from .db import connection, now_iso


def init_study_db():
    """初始化学习模式数据表（含第一期 analytics 与 events 新列）。"""
    with connection() as c:
        c.execute('''CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY,
            person_id INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            study_seconds INTEGER DEFAULT 0,
            break_seconds INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS study_events (
            id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            event_data TEXT,
            created_at TEXT NOT NULL
        )''')
        # 第一期：事件质量自评列（旧库迁移，重复执行安全）
        for ddl in ("ALTER TABLE study_events ADD COLUMN quality_score INTEGER",
                    "ALTER TABLE study_events ADD COLUMN context_tags TEXT"):
            try:
                c.execute(ddl)
            except Exception:
                pass  # 列已存在
        c.execute('''CREATE TABLE IF NOT EXISTS study_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            period_type TEXT NOT NULL,
            total_study_minutes INTEGER NOT NULL,
            total_break_minutes INTEGER NOT NULL,
            total_sessions INTEGER NOT NULL,
            completed_sessions INTEGER NOT NULL,
            interrupted_sessions INTEGER NOT NULL,
            avg_focus_minutes REAL,
            max_focus_minutes INTEGER,
            extend_count INTEGER,
            early_break_count INTEGER,
            best_time_slot TEXT,
            worst_time_slot TEXT,
            distraction_triggers TEXT,
            recovery_speed_seconds INTEGER,
            created_at TEXT NOT NULL,
            UNIQUE(person_id, period_start, period_type)
        )''')


def start_session(person_id: int) -> dict:
    """开始新学习会话。同一人同时只能有一个活动会话。"""
    with connection() as c:
        active = c.execute(
            "SELECT id FROM study_sessions WHERE person_id=? AND status='active'",
            (person_id,)
        ).fetchone()
        if active:
            raise ValueError("已经在学习中，请先结束当前会话。")
        cur = c.execute(
            "INSERT INTO study_sessions(person_id,started_at) VALUES(?,?)",
            (person_id, now_iso())
        )
        session_id = cur.lastrowid
        c.execute(
            "INSERT INTO study_events(session_id,event_type,created_at) VALUES(?,'start',?)",
            (session_id, now_iso())
        )
    return get_session(session_id)


def get_session(session_id: int) -> dict | None:
    """获取会话详情。"""
    with connection() as c:
        row = c.execute(
            "SELECT * FROM study_sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def get_active_session(person_id: int) -> dict | None:
    """获取活动（active）会话。"""
    with connection() as c:
        row = c.execute(
            "SELECT * FROM study_sessions WHERE person_id=? AND status='active'",
            (person_id,)
        ).fetchone()
        return dict(row) if row else None


def get_current_session(person_id: int) -> dict | None:
    """获取当前承载中的会话（active 或 paused）。

    离座自动暂停后，页面卡片、坐姿守护、结束/自评都必须还能找回这条会话；
    只对"提醒循环"来说 paused 不算活动，其余场景一律用本函数。
    """
    with connection() as c:
        row = c.execute(
            "SELECT * FROM study_sessions WHERE person_id=? AND status IN ('active','paused') "
            "ORDER BY id DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        return dict(row) if row else None


def end_session(session_id: int) -> dict:
    """结束会话，并把学习/休息总时长持久化到会话行（第一期：供画像分析）。"""
    sess = get_session(session_id)
    if not sess:
        raise ValueError("学习会话不存在")
    elapsed = compute_elapsed(sess)
    with connection() as c:
        c.execute(
            "UPDATE study_sessions SET status='completed',ended_at=?,study_seconds=?,break_seconds=? WHERE id=?",
            (now_iso(), elapsed['study_seconds'], elapsed['break_seconds'], session_id)
        )
        c.execute(
            "INSERT INTO study_events(session_id,event_type,created_at) VALUES(?,'end',?)",
            (session_id, now_iso())
        )
    return get_session(session_id)


def rate_session(session_id: int, score: int, tags: list[str] | None = None) -> dict:
    """用户自评本次学习质量（1-10），可带场景标签。记为带质量分的事件。"""
    if not 1 <= score <= 10:
        raise ValueError("评分需在1-10之间")
    tag_text = ",".join(t.strip() for t in (tags or []) if t.strip())[:200] or None
    with connection() as c:
        if not c.execute("SELECT id FROM study_sessions WHERE id=?", (session_id,)).fetchone():
            raise ValueError("学习会话不存在")
        c.execute(
            "INSERT INTO study_events(session_id,event_type,event_data,quality_score,context_tags,created_at) "
            "VALUES(?,'rate',?,?,?,?)",
            (session_id, f"score:{score}", score, tag_text, now_iso())
        )
    return {"ok": True, "session_id": session_id, "score": score, "tags": tag_text}


def pause_session(session_id: int) -> dict:
    """暂停学习（进入休息）。"""
    with connection() as c:
        c.execute(
            "UPDATE study_sessions SET status='paused' WHERE id=?", (session_id,)
        )
        c.execute(
            "INSERT INTO study_events(session_id,event_type,created_at) VALUES(?,'break',?)",
            (session_id, now_iso())
        )
    return get_session(session_id)


def resume_session(session_id: int) -> dict:
    """恢复学习。"""
    with connection() as c:
        c.execute(
            "UPDATE study_sessions SET status='active' WHERE id=?", (session_id,)
        )
        c.execute(
            "INSERT INTO study_events(session_id,event_type,created_at) VALUES(?,'resume',?)",
            (session_id, now_iso())
        )
    return get_session(session_id)


def record_reminder(session_id: int, reminder_type: str, data: str = None):
    """记录主动提醒事件。"""
    with connection() as c:
        c.execute(
            "INSERT INTO study_events(session_id,event_type,event_data,created_at) VALUES(?,'reminder',?,?)",
            (session_id, f"{reminder_type}:{data or ''}", now_iso())
        )


def compute_elapsed(session: dict, now: datetime = None) -> dict:
    """计算已学习时长和已休息时长（秒）。
    
    返回 {study_seconds, break_seconds, current_phase, phase_elapsed_seconds}
    """
    if not session:
        return {"study_seconds": 0, "break_seconds": 0, "current_phase": None, "phase_elapsed_seconds": 0}
    
    now = now or datetime.now().astimezone()
    started = datetime.fromisoformat(session['started_at'])
    
    with connection() as c:
        events = c.execute(
            "SELECT event_type,created_at FROM study_events WHERE session_id=? ORDER BY id",
            (session['id'],)
        ).fetchall()
    
    study_total = 0
    break_total = 0
    current_phase = 'study' if session['status'] == 'active' else 'break'
    phase_start = started
    
    for evt in events:
        evt_type = evt[0]
        evt_time = datetime.fromisoformat(evt[1])
        if evt_type == 'break':
            # 之前在学习，现在进入休息
            study_total += (evt_time - phase_start).total_seconds()
            phase_start = evt_time
            current_phase = 'break'
        elif evt_type == 'resume':
            # 之前在休息，现在恢复学习
            break_total += (evt_time - phase_start).total_seconds()
            phase_start = evt_time
            current_phase = 'study'
    
    # 加上当前阶段到现在的时长
    phase_elapsed = (now - phase_start).total_seconds()
    if current_phase == 'study':
        study_total += phase_elapsed
    else:
        break_total += phase_elapsed
    
    return {
        "study_seconds": int(study_total),
        "break_seconds": int(break_total),
        "current_phase": current_phase if session['status'] != 'completed' else None,
        "phase_elapsed_seconds": int(phase_elapsed)
    }


def last_extension(session_id: int) -> tuple[int, datetime] | None:
    """返回最近一次"再给我N分钟"的 (分钟数, 请求时刻)，没有则 None。"""
    with connection() as c:
        row = c.execute(
            "SELECT event_data,created_at FROM study_events "
            "WHERE session_id=? AND event_type='reminder' AND event_data LIKE 'extend:%' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row:
        return None
    try:
        minutes = int((row[0] or '').split(':', 1)[1])
    except (ValueError, IndexError):
        minutes = 5
    return minutes, datetime.fromisoformat(row[1])


def _phase_start(session: dict, now: datetime) -> datetime:
    """当前阶段（学习/休息）的起点：最近一次 break/resume 事件时间，没有则会话开始时间。"""
    with connection() as c:
        row = c.execute(
            "SELECT created_at FROM study_events WHERE session_id=? AND event_type IN ('break','resume') "
            "ORDER BY id DESC LIMIT 1",
            (session['id'],),
        ).fetchone()
    base = datetime.fromisoformat(row[0]) if row else datetime.fromisoformat(session['started_at'])
    return base


def _gentled_after(session: dict, phase_start: datetime) -> bool:
    """本阶段是否已经柔性提醒过（避免重复骚扰）。"""
    with connection() as c:
        row = c.execute(
            "SELECT id FROM study_events WHERE session_id=? AND event_type='reminder' "
            "AND event_data LIKE 'gentle:%' AND created_at>=? LIMIT 1",
            (session['id'], phase_start.isoformat()),
        ).fetchone()
    return row is not None


def sessions_needing_reminder(focus_minutes: int, break_minutes: int,
                              threshold_resolver=None, now: datetime = None) -> list[dict]:
    """返回需要主动行动的会话列表（第一期：自适应）。

    每项 {'session', 'type', 'elapsed', 'spoken'}，type：
    - 'gentle'：到点前 5 分钟柔性确认（仅对历史上常延长的孩子）
    - 'break_time'：专注到点，提醒休息并自动进入休息
    - 'resume_time'：休息到点，提醒回来继续并自动恢复

    threshold_resolver(person_id) -> {'focus_minutes','break_minutes','extend_prone'}；
    不提供时用固定参数（第 0 期行为）。延长语义：'再给我N分钟'把到点时间推为
    请求时刻+N分钟，而不是修改阈值——到期后若再次延长，再向后推。
    """
    now = now or datetime.now().astimezone()
    with connection() as c:
        sessions = c.execute(
            "SELECT * FROM study_sessions WHERE status IN ('active','paused')"
        ).fetchall()

    result = []
    for row in sessions:
        sess = dict(row)
        pid = sess['person_id']
        limits = {'focus_minutes': focus_minutes, 'break_minutes': break_minutes, 'extend_prone': False}
        if threshold_resolver:
            try:
                override = threshold_resolver(pid)
                if override:
                    limits.update({k: v for k, v in override.items() if v is not None})
            except Exception:
                pass

        elapsed = compute_elapsed(sess, now)
        phase = elapsed['current_phase']
        if phase not in ('study', 'break'):
            continue
        base_minutes = limits['focus_minutes'] if phase == 'study' else limits['break_minutes']
        phase_start = _phase_start(sess, now)
        due = phase_start + timedelta(minutes=base_minutes)

        extend = last_extension(sess['id'])
        if extend and extend[1] >= phase_start:
            extended_due = extend[1] + timedelta(minutes=extend[0])
            if extended_due > due:
                due = extended_due

        if now >= due:
            spoken = (f"学了 {base_minutes} 分钟了，起来活动 {limits['break_minutes']} 分钟吧。"
                      if phase == 'study' else "休息够了，继续加油吧！")
            result.append({'session': sess, 'type': 'break_time' if phase == 'study' else 'resume_time',
                           'elapsed': elapsed, 'spoken': spoken})
        elif phase == 'study' and limits['extend_prone'] and not _gentled_after(sess, phase_start):
            gentle_at = due - timedelta(minutes=5)
            if now >= gentle_at:
                minutes_left = max(1, int((due - now).total_seconds() // 60) + 1)
                spoken = f"还有大约 {minutes_left} 分钟到休息时间，要继续还是先休息？"
                result.append({'session': sess, 'type': 'gentle', 'elapsed': elapsed, 'spoken': spoken})

    return result


def compute_period_stats(person_id: int, period_start: datetime, period_end: datetime,
                         period_type: str = 'weekly', save: bool = True) -> dict:
    """聚合指定周期的学习数据并写入 study_analytics 快照。"""
    ps, pe = period_start.isoformat(), period_end.isoformat()
    with connection() as c:
        sessions = [dict(r) for r in c.execute(
            "SELECT * FROM study_sessions WHERE person_id=? AND started_at>=? AND started_at<?",
            (person_id, ps, pe)).fetchall()]
        completed = [s for s in sessions if s['status'] == 'completed']
        interrupted = [s for s in sessions if s['status'] not in ('completed',)]
        ids = [s['id'] for s in sessions]
        placeholders = ','.join('?' * len(ids)) or '-1'
        events = [dict(r) for r in c.execute(
            f"SELECT * FROM study_events WHERE session_id IN ({placeholders})", ids).fetchall()] if ids else []

        focus_list = [s['study_seconds'] for s in completed if s['study_seconds']]
        recover_pairs = []
        breaks_by_session: dict[int, list] = {}
        for e in events:
            if e['event_type'] == 'break':
                breaks_by_session.setdefault(e['session_id'], []).append(e)
            elif e['event_type'] == 'resume':
                prev = breaks_by_session.get(e['session_id'])
                if prev:
                    b = prev.pop()
                    recover_pairs.append((datetime.fromisoformat(e['created_at'])
                                          - datetime.fromisoformat(b['created_at'])).total_seconds())

        extend_count = sum(1 for e in events if (e['event_data'] or '').startswith('extend:'))
        early_breaks = sum(1 for e in events if e['event_type'] == 'break')
        tags = [t for e in events if e['event_type'] == 'rate' and e.get('context_tags')
                for t in e['context_tags'].split(',')]
        tag_counts: dict[str, int] = {}
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        triggers = sorted(tag_counts, key=tag_counts.get, reverse=True)[:3] or None

        # 时段分布：按完成会话开始小时统计学习分钟数
        hour_load: dict[int, int] = {}
        for s in completed:
            h = datetime.fromisoformat(s['started_at']).hour
            hour_load[h] = hour_load.get(h, 0) + s['study_seconds'] // 60

        def slot(h: int) -> str:
            return f"{h:02d}:00-{h + 1:02d}:00"
        best = max(hour_load, key=hour_load.get) if hour_load else None
        worst = None
        if len(hour_load) >= 3:
            worst = min(hour_load, key=hour_load.get)
        elif interrupted and not completed:
            worst = datetime.fromisoformat(interrupted[0]['started_at']).hour

        stats = {
            "person_id": person_id,
            "period_start": ps, "period_end": pe, "period_type": period_type,
            "total_study_minutes": sum(s['study_seconds'] for s in sessions) // 60,
            "total_break_minutes": sum(s['break_seconds'] for s in sessions) // 60,
            "total_sessions": len(sessions),
            "completed_sessions": len(completed),
            "interrupted_sessions": len(interrupted),
            "avg_focus_minutes": round(sum(focus_list) / len(focus_list) / 60, 1) if focus_list else None,
            "max_focus_minutes": max(focus_list) // 60 if focus_list else None,
            "extend_count": extend_count,
            "early_break_count": early_breaks,
            "best_time_slot": slot(best) if best is not None else None,
            "worst_time_slot": slot(worst) if worst is not None else None,
            "distraction_triggers": triggers,
            "recovery_speed_seconds": round(sum(recover_pairs) / len(recover_pairs)) if recover_pairs else None,
        }
        if save:
            with connection() as c:
                c.execute(
                    "INSERT OR REPLACE INTO study_analytics(person_id,period_start,period_end,period_type,"
                    "total_study_minutes,total_break_minutes,total_sessions,completed_sessions,interrupted_sessions,"
                    "avg_focus_minutes,max_focus_minutes,extend_count,early_break_count,best_time_slot,worst_time_slot,"
                    "distraction_triggers,recovery_speed_seconds,created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (person_id, ps, pe, period_type, stats["total_study_minutes"], stats["total_break_minutes"],
                     stats["total_sessions"], stats["completed_sessions"], stats["interrupted_sessions"],
                     stats["avg_focus_minutes"], stats["max_focus_minutes"], stats["extend_count"],
                     stats["early_break_count"], stats["best_time_slot"], stats["worst_time_slot"],
                     str(triggers) if triggers else None, stats["recovery_speed_seconds"], now_iso()))
    return stats
