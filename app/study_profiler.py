"""学习画像生成器（第一期核心）：从历史数据发现个人学习规律。

冷启动策略（设计文档 §5.1）：
- 完成会话 < 3 个：全部使用配置默认值（通用节奏）；
- >= 3 个：用历史均值优化专注时长；
- >= 5 个：加上时段偏好与延长倾向判断。
"""
from collections import Counter
from datetime import datetime, timedelta
from .db import connection
from . import study
from .config import settings

MIN_SESSIONS_FOR_AVG = 3      # 达到此完成会话数才开始用均值调整时长
MIN_SESSIONS_FOR_PATTERN = 5  # 达到此数才判断延长倾向/时段偏好


def recent_sessions(person_id: int, days: int = 30) -> list[dict]:
    since = (datetime.now().astimezone() - timedelta(days=days)).isoformat()
    with connection() as c:
        return [dict(r) for r in c.execute(
            "SELECT * FROM study_sessions WHERE person_id=? AND started_at>=?",
            (person_id, since)).fetchall()]


def _session_focus_seconds(session: dict) -> int:
    """完成会话优先用持久化列；旧数据没有时回退到事件重算。"""
    if session.get('study_seconds'):
        return int(session['study_seconds'])
    return study.compute_elapsed(session)['study_seconds']


def build_profile(person_id: int, days: int = 30) -> dict:
    """生成学习画像。字段含义见设计文档 §2.2 模块A。"""
    sessions = recent_sessions(person_id, days)
    completed = [s for s in sessions if s['status'] == 'completed']
    now = datetime.now().astimezone()

    # 最佳专注时长：完成会话均值四舍五入到 5 分钟，限制在 10~90 分钟
    focus_values = [_session_focus_seconds(s) for s in completed if _session_focus_seconds(s) > 60]
    default_focus = settings.study_focus_minutes
    if len(completed) >= MIN_SESSIONS_FOR_AVG and focus_values:
        avg = sum(focus_values) / len(focus_values) / 60
        optimal = max(10, min(90, int(round(avg / 5) * 5)))
    else:
        optimal = default_focus

    # 时段偏好：完成会话最多的开始小时
    hour_counter = Counter(datetime.fromisoformat(s['started_at']).hour for s in completed)
    best_hours = [h for h, n in hour_counter.most_common(3) if n >= 2] if len(completed) >= MIN_SESSIONS_FOR_PATTERN else []

    # 恢复速度：break→resume 平均间隔
    recovery = None
    ids = [s['id'] for s in sessions]
    if ids:
        with connection() as c:
            placeholders = ','.join('?' * len(ids))
            events = c.execute(
                f"SELECT session_id,event_type,created_at FROM study_events "
                f"WHERE session_id IN ({placeholders}) AND event_type IN ('break','resume') ORDER BY id",
                ids).fetchall()
        pending: dict[int, datetime] = {}
        pairs = []
        for e in events:
            if e[1] == 'break':
                pending[e[0]] = datetime.fromisoformat(e[2])
            elif e[1] == 'resume' and e[0] in pending:
                pairs.append((datetime.fromisoformat(e[2]) - pending.pop(e[0])).total_seconds())
        if pairs:
            recovery = round(sum(pairs) / len(pairs) / 60)

    # 延长倾向：extend 事件比例
    extend_events = 0
    if ids:
        with connection() as c:
            placeholders = ','.join('?' * len(ids))
            extend_events = c.execute(
                f"SELECT COUNT(*) FROM study_events WHERE session_id IN ({placeholders}) "
                f"AND event_data LIKE 'extend:%'", ids).fetchone()[0]
    extend_prone = (len(completed) >= MIN_SESSIONS_FOR_PATTERN
                    and extend_events / max(len(completed), 1) > 0.3)

    # 常见分心/状态标签（来自自评事件）
    distraction = []
    if ids:
        with connection() as c:
            placeholders = ','.join('?' * len(ids))
            rows = c.execute(
                f"SELECT context_tags FROM study_events WHERE session_id IN ({placeholders}) "
                f"AND event_type='rate' AND context_tags IS NOT NULL", ids).fetchall()
        tags = [t.strip() for r in rows for t in (r[0] or '').split(',') if t.strip()]
        distraction = [t for t, n in Counter(tags).most_common(3) if n >= 2]

    completion_rate = round(len(completed) / len(sessions), 2) if sessions else None

    # 趋势：最近 7 天 vs 前 7 天的学习分钟数
    trend = 'new'
    if sessions:
        recent = sum(_session_focus_seconds(s) for s in sessions
                     if datetime.fromisoformat(s['started_at']) >= now - timedelta(days=7))
        prior = sum(_session_focus_seconds(s) for s in sessions
                    if now - timedelta(days=14) <= datetime.fromisoformat(s['started_at']) < now - timedelta(days=7))
        if prior > 0 and recent / prior > 1.1:
            trend = 'improving'
        elif prior > 0 and recent / prior < 0.9:
            trend = 'declining'
        else:
            trend = 'stable'

    return {
        "person_id": person_id,
        "optimal_focus_minutes": optimal,
        "break_minutes": settings.study_break_minutes,
        "best_hours": best_hours,
        "avg_recovery_minutes": recovery,
        "distraction_patterns": distraction,
        "completion_rate": completion_rate,
        "extend_prone": extend_prone,
        "trend": trend,
        "sessions_analyzed": len(completed),
    }


def thresholds_for_reminder(person_id: int) -> dict:
    """给 study_loop 用的自适应阈值解析器。"""
    profile = build_profile(person_id, days=14)
    return {
        'focus_minutes': profile['optimal_focus_minutes'],
        'break_minutes': profile['break_minutes'],
        'extend_prone': profile['extend_prone'],
    }


def suggest(person_id: int) -> dict:
    """给用户/Agent 的可执行建议（设计文档验收项：至少 1 条）。"""
    profile = build_profile(person_id)
    suggestions = []
    if profile['sessions_analyzed'] < MIN_SESSIONS_FOR_AVG:
        suggestions.append(f"再多用几次，我就能找到属于你的节奏（已分析 {profile['sessions_analyzed']} 次完整学习）。")
    else:
        suggestions.append(f"你的最佳专注时长约 {profile['optimal_focus_minutes']} 分钟，已按此调整提醒。")
    if profile['best_hours']:
        slots = "、".join(f"{h}点前后" for h in profile['best_hours'])
        suggestions.append(f"你的高效时段在 {slots}，难的任务可以放这里。")
    if profile['avg_recovery_minutes'] and profile['avg_recovery_minutes'] > settings.study_break_minutes:
        suggestions.append(f"你平均需要 {profile['avg_recovery_minutes']} 分钟才能缓过来，休息可以再长一点。")
    if profile['trend'] == 'declining':
        suggestions.append("这周学习时间比上周少，别自责，安排一次短学习把节奏找回来。")
    return {'profile': profile, 'suggestions': suggestions}
