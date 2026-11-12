"""行为洞察（第二期 2B）：从坐姿守护的数值事件中提取行为规律。

只读聚合，不改数据；输出用于报告与 study_suggest。
"""
from datetime import datetime, timedelta
from .db import connection


def analyze_behavior_pattern(person_id: int, days: int = 7) -> dict:
    since = (datetime.now().astimezone() - timedelta(days=days)).isoformat()
    try:
        with connection() as c:
            rows = c.execute(
                "SELECT posture_quality,issues,warning_sent,user_responded,present FROM study_vision_events "
                "WHERE person_id=? AND created_at>=?", (person_id, since)).fetchall()
    except Exception:
        return {"has_data": False, "frames": 0}  # 表未建（未启用视觉）时优雅降级
    if not rows:
        return {"has_data": False, "frames": 0}

    total = len(rows)
    present_frames = sum(1 for r in rows if r[4])
    good = sum(1 for r in rows if r[0] == 'good')
    poor = sum(1 for r in rows if r[0] == 'poor')
    absent_ratio = 1 - present_frames / total
    warnings = sum(1 for r in rows if r[2])
    responded = sum(1 for r in rows if r[2] and r[3])

    issue_counts: dict[str, int] = {}
    for r in rows:
        for i in (r[1] or '').split(','):
            if i:
                issue_counts[i] = issue_counts.get(i, 0) + 1
    common = sorted(issue_counts, key=issue_counts.get, reverse=True)[:2]

    return {
        "has_data": True,
        "frames": total,
        "days": days,
        "posture_good_ratio": round(good / max(present_frames, 1), 2),
        "poor_frames": poor,
        "absent_ratio": round(absent_ratio, 2),
        "warnings_sent": warnings,
        "warning_response_rate": round(responded / warnings, 2) if warnings else None,
        "common_issues": common,
    }


def behavior_suggestions(person_id: int) -> list[str]:
    p = analyze_behavior_pattern(person_id)
    out = []
    if not p['has_data']:
        return out
    if p['common_issues']:
        out.append(f"最常见的坐姿问题是：{'、'.join(p['common_issues'])}，可以先只改这一项")
    if p['warning_response_rate'] is not None and p['warnings_sent'] >= 3:
        rate = round(p['warning_response_rate'] * 100)
        if rate >= 70:
            out.append(f"提醒后你自己就坐好了（{rate}% 的提醒都改正了），说明坐姿意识在建立")
        else:
            out.append(f"提醒后改正率 {rate}%，可以试试每 20 分钟主动站起来一次，比被提醒更省力")
    if p['absent_ratio'] > 0.4:
        out.append("学习时段里经常不在座位，计时会自动暂停，回来后继续，不用担心被算多")
    return out
