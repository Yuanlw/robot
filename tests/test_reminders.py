import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timedelta
from app.db import init_db, connection
from app.commands import parse_reminder
from app.reminders import create, due_now, get, mark_delivered, update

class ReminderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='robot-reminder-test-')
        self.addCleanup(temporary.cleanup)
        db_patch = patch('app.db.DB_PATH', Path(temporary.name) / 'robot.db')
        db_patch.start()
        self.addCleanup(db_patch.stop)
        init_db()

    def test_same_instant_across_timezones(self):
        item=create('测试',datetime.fromisoformat('2026-10-27T12:30:00+00:00'))
        self.assertEqual(due_now(datetime.fromisoformat('2026-10-27T20:29:59+08:00')),[])
        self.assertEqual([x['id'] for x in due_now(datetime.fromisoformat('2026-10-27T20:30:00+08:00'))],[item['id']])

    def test_reading_due_does_not_acknowledge_delivery(self):
        item=create('测试',datetime.fromisoformat('2020-01-01T00:00:00+08:00'))
        self.assertEqual(due_now(expiry_hours=None)[0]['id'],item['id'])
        self.assertEqual(get(item['id'])['status'],'active')
        mark_delivered(item['id'])
        self.assertEqual(due_now(expiry_hours=None),[])

    def test_half_hour_is_time_not_reminder_content(self):
        text,due=parse_reminder('提醒我明天晚上八点半练琴')
        self.assertEqual((text,due.hour,due.minute),('练琴',20,30))

    def test_explicit_early_morning_reminder_fires(self):
        # 2026-11-13 回归：安静时段不得拦截用户明确设定的提醒——7点的药就该7点响。
        # 安静时段只约束机器人自主播报（study_loop 内），不在 due_now 里。
        early = datetime.now().astimezone().replace(hour=6, minute=30, second=0, microsecond=0)
        item = create('吃药', early)
        self.assertEqual([x['id'] for x in due_now(early)], [item['id']])

    def test_expired_reminder_not_replayed(self):
        # 陈旧补播由 24h 过期兜底：30 小时前的提醒不再自动播放
        old = datetime.now().astimezone() - timedelta(hours=30)
        create('过期提醒', old)
        self.assertEqual(due_now(datetime.now().astimezone()), [])


class RepeatReminderTests(unittest.TestCase):
    """P0-4 重复提醒：送达后推进到下一次，而非终结；静音/停机不丢。"""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='robot-repeat-test-')
        self.addCleanup(temporary.cleanup)
        db_patch = patch('app.db.DB_PATH', Path(temporary.name) / 'robot.db')
        db_patch.start()
        self.addCleanup(db_patch.stop)
        init_db()

    def test_daily_advances_one_day_and_stays_active(self):
        due = datetime.fromisoformat('2026-11-16T20:00:00+08:00')
        item = create('练琴', due, repeat_rule='daily')
        mark_delivered(item['id'], now=due + timedelta(minutes=1))
        after = get(item['id'])
        self.assertEqual(after['status'], 'active')
        self.assertEqual(after['due_at'], '2026-11-17T20:00:00+08:00')
        self.assertEqual(after['repeat_rule'], 'daily')

    def test_daily_skips_missed_occurrences(self):
        due = datetime.fromisoformat('2026-11-16T20:00:00+08:00')
        item = create('练琴', due, repeat_rule='daily')
        # 静音/停机 3 天后恢复：只推进到下一个未来时刻，不补播陈旧
        mark_delivered(item['id'], now=due + timedelta(days=3, hours=1))
        self.assertEqual(get(item['id'])['due_at'], '2026-11-20T20:00:00+08:00')

    def test_weekly_advances_seven_days(self):
        due = datetime.fromisoformat('2026-11-16T09:00:00+08:00')
        item = create('开会', due, repeat_rule='weekly')
        mark_delivered(item['id'], now=due + timedelta(minutes=1))
        self.assertEqual(get(item['id'])['due_at'], '2026-11-23T09:00:00+08:00')

    def test_once_still_terminal(self):
        due = datetime.fromisoformat('2026-11-16T20:00:00+08:00')
        item = create('单次', due, repeat_rule='once')
        mark_delivered(item['id'], now=due + timedelta(minutes=1))
        self.assertEqual(get(item['id'])['status'], 'delivered')

    def test_repeating_exempt_from_expiry_drop(self):
        # 重复提醒即使过期超过 24h 也不被丢弃（保持 active 等待重试/推进）
        old = datetime.now().astimezone() - timedelta(days=5)
        item = create('每天吃药', old, repeat_rule='daily')
        self.assertIn(item['id'], [x['id'] for x in due_now(datetime.now().astimezone())])

    def test_update_repeat_rule_and_preserve_when_omitted(self):
        due = datetime.now().astimezone() + timedelta(days=1)
        item = create('练琴', due, repeat_rule='daily')
        changed = update(item['id'], '练琴', due, repeat_rule='weekly')
        self.assertEqual(changed['repeat_rule'], 'weekly')
        preserved = update(item['id'], '练琴改名', due)  # 不传 repeat → 保留 weekly
        self.assertEqual(preserved['repeat_rule'], 'weekly')
        self.assertEqual(preserved['text'], '练琴改名')

    def test_invalid_repeat_rejected(self):
        due = datetime.now().astimezone() + timedelta(days=1)
        with self.assertRaises(ValueError):
            create('x', due, repeat_rule='hourly')
