import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import settings


CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def number(value: str) -> int:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, right = value.split("十", 1)
        return (CN_NUM.get(left, 1) * 10) + CN_NUM.get(right, 0)
    return CN_NUM.get(value, 0)


def parse_reminder(command: str) -> tuple[str, datetime] | None:
    match = re.search(r"提醒我(?P<day>今天|明天|后天)?(?P<period>早上|上午|中午|下午|晚上)?(?P<hour>[0-9一二两三四五六七八九十]{1,3})点(?P<minute>[0-9一二两三四五六七八九十]{1,3})?分?(?P<text>.+)", command.strip())
    if not match:
        return None
    now = datetime.now(ZoneInfo(settings.timezone))
    day = {"明天": 1, "后天": 2}.get(match.group("day"), 0)
    hour = number(match.group("hour"))
    minute = number(match.group("minute")) if match.group("minute") else 0
    if match.group("period") in {"下午", "晚上"} and hour < 12:
        hour += 12
    due = (now + timedelta(days=day)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        due += timedelta(days=1)
    return match.group("text").strip(), due
