import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from datetime import datetime
from app.db import init_db, connection
from app.commands import parse_reminder
from app.reminders import create, due_now, get, mark_delivered

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
