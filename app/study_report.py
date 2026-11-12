"""学习报告生成（第一期）：周/月 Markdown 报告。

数据来源：study.compute_period_stats（周期聚合）+ study_profiler（画像）。
隐私：只用时长/次数/时段等聚合数字，不含任何录音文本内容。
飞书推送不在本期范围：先通过 API/Agent 返回 Markdown 文本。
"""
from datetime import datetime, timedelta
from . import study, study_profiler
from .config import settings


def _fmt_minutes(minutes: int) -> str:
    h, m = divmod(int(minutes), 60)
    return f"{h}小时{m}分钟" if h else f"{m}分钟"


def _pct(value) -> str:
    return f"{round(value * 100)}%" if value is not None else "—"


def build_report(person_id: int, person_name: str, period: str = 'week') -> dict:
    """生成周期报告。period: today | week | month。"""
    now = datetime.now().astimezone()
    if period == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now + timedelta(seconds=1)
        period_type, title_suffix = 'daily', "今日"
    elif period == 'month':
        start = now - timedelta(days=30)
        end = now + timedelta(seconds=1)
        period_type, title_suffix = 'monthly', "近30天"
    else:
        start = now - timedelta(days=7)
        end = now + timedelta(seconds=1)
        period_type, title_suffix = 'weekly', "本周"

    stats = study.compute_period_stats(person_id, start, end, period_type)
    profile = study_profiler.build_profile(person_id, days=30 if period != 'today' else 7)

    lines = [f"# {person_name}的学习报告（{title_suffix}）", ""]
    lines += [
        "## 成绩单", "",
        f"- 📚 总学习时长：**{_fmt_minutes(stats['total_study_minutes'])}**",
        f"- ✅ 完成会话：**{stats['completed_sessions']} 次**（完成率 {_pct(stats['completed_sessions'] / stats['total_sessions']) if stats['total_sessions'] else '—'}）",
        f"- 🔥 最长专注：**{stats['max_focus_minutes']} 分钟**" if stats['max_focus_minutes'] else "- 🔥 还没有完整专注记录",
        f"- ⚡ 平均专注：**{stats['avg_focus_minutes']} 分钟**" if stats['avg_focus_minutes'] else "- ⚡ 平均专注：数据积累中",
        f"- 🔄 延长 {stats['extend_count']} 次 · 休息 {stats['early_break_count']} 次",
        "",
    ]

    insights = []
    if stats['best_time_slot']:
        insights.append(f"- ✨ 最高效时段：**{stats['best_time_slot']}**，重要的学习可以放这里")
    if stats['worst_time_slot'] and stats['worst_time_slot'] != stats['best_time_slot']:
        insights.append(f"- ⚠️ {stats['worst_time_slot']} 相对容易分心，可以缩短时长或降低任务难度")
    if profile['avg_recovery_minutes']:
        insights.append(f"- 💤 你平均休息 {profile['avg_recovery_minutes']} 分钟能恢复状态")
    if profile['distraction_patterns']:
        insights.append(f"- 🧩 常见状态标签：{'、'.join(profile['distraction_patterns'])}")
    if insights:
        lines += ["## 发现的规律", "", *insights, ""]

    next_steps = []
    if profile['optimal_focus_minutes'] != settings.study_focus_minutes:
        next_steps.append(f"提醒时长已按你的节奏调到 {profile['optimal_focus_minutes']} 分钟")
    if profile['trend'] == 'improving':
        next_steps.append("本周比上周投入更多，继续保持这个势头")
    elif profile['trend'] == 'declining':
        next_steps.append("这周投入比上周少，明天安排一次短学习把节奏找回来")
    if stats['total_sessions'] == 0:
        next_steps.append("这个周期还没有学习记录，随时说一句\u201c开始学习\u201d就行")
    if next_steps:
        lines += ["## 下一步建议", "", *[f"1. {s}" for s in next_steps], ""]

    lines += ["---", "_数据来源：阿伴学习陪伴系统（仅统计时长与次数，不含学习内容）_"]

    trend_cn = {'improving': '上升', 'stable': '平稳', 'declining': '回落', 'new': '新开始'}.get(profile['trend'], profile['trend'])
    return {
        "period": period,
        "markdown": "\n".join(lines),
        "stats": stats,
        "profile": profile,
        "headline": f"{title_suffix}学习 {_fmt_minutes(stats['total_study_minutes'])}，{len(insights)} 条规律，趋势{trend_cn}",
    }
