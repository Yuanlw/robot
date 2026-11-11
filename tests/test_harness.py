import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import agent_store, db, reminders
from app.agent_tools import execute
from app.harness import HarnessService, explicit_confirmation, agent
from app.config import settings
from app.main import app


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p = patch.object(db, 'DB_PATH', Path(tmp.name) / 'robot.db')
        p.start(); self.addCleanup(p.stop)
        db.init_db(); agent_store.init_agent_db()
        self.sid = agent_store.session('browser-a')['harness_id']
        self.at = datetime.now().astimezone()
        self.context = {'session_id': self.sid, 'received_at': self.at.isoformat(), 'confirmations': {}}
        self.ref = execute('get_current_time', {}, self.context)['reference_id']

    def draft(self):
        return execute('prepare_reminder', {'text': '削笔', 'delay_seconds': 120, 'reference_id': self.ref}, self.context)

    def test_relative_time_and_content_only_edit(self):
        row = self.draft()
        self.assertEqual(datetime.fromisoformat(row['due_at']), self.at + timedelta(seconds=120))
        self.context['received_at'] = (self.at + timedelta(seconds=10)).isoformat()
        edited = execute('prepare_reminder', {'draft_id': row['id'], 'text': '喝水'}, self.context)
        self.assertEqual(edited['due_at'], row['due_at'])
        edited = execute('prepare_reminder', {'draft_id': row['id'], 'delay_seconds': 60, 'reference_id': self.ref}, self.context)
        self.assertEqual(datetime.fromisoformat(edited['due_at']), self.at + timedelta(seconds=70))

    def test_draft_not_scheduled_and_confirm_exactly_once(self):
        row = self.draft()
        self.assertEqual(reminders.list_active(), [])
        with self.assertRaises(ValueError):
            execute('confirm_reminder', {'id': row['id'], 'revision': 1}, self.context)
        self.context['confirmations'] = {row['id']: 1}
        a = execute('confirm_reminder', {'id': row['id'], 'revision': 1}, self.context)
        b = execute('confirm_reminder', {'id': row['id'], 'revision': 1}, self.context)
        self.assertEqual(a['id'], b['id'])
        self.assertEqual(len(reminders.list_active()), 1)
        self.assertEqual(reminders.due_now(self.at + timedelta(seconds=121))[0]['id'], a['id'])
        reminders.mark_delivered(a['id'])
        self.assertEqual(reminders.list_active(), [])

    def test_stale_confirmation_and_other_session_rejected(self):
        row = self.draft()
        agent_store.prepare(self.sid, {'draft_id': row['id'], 'text': '喝水'}, self.at.isoformat())
        with self.assertRaises(ValueError):
            agent_store.confirm(self.sid, row['id'], 1)
        with self.assertRaises(ValueError):
            agent_store.confirm('another-session', row['id'], 2)
        self.assertEqual(reminders.list_active(), [])

    def test_missing_content_cross_session_clock_and_bad_timezone(self):
        for args in ({'text': '提醒', 'delay_seconds': 120, 'reference_id': self.ref},
                     {'text': '喝水', 'due_at': '2099-01-01T12:00:00'}):
            with self.assertRaises(ValueError):
                execute('prepare_reminder', args, self.context)
        with self.assertRaises(ValueError):
            execute('prepare_reminder', {'text': '喝水', 'delay_seconds': 120, 'reference_id': self.ref},
                    {**self.context, 'session_id': 'other'})

    def test_confirm_language_does_not_authorize_corrections(self):
        for text in ['确认', '确认，就这样。', '好的']:
            self.assertTrue(explicit_confirmation(text))
        for text in ['不要确认', '确认，但是改成明天', '如果我说确认会怎么样', '"确认"']:
            self.assertFalse(explicit_confirmation(text))

    def test_edit_active_reminder_keeps_original_until_confirmation(self):
        row = self.draft()
        original = agent_store.confirm(self.sid, row['id'], row['revision'])
        draft = execute('prepare_reminder', {'reminder_id': original['id'], 'text': '喝水'}, self.context)
        self.assertEqual(reminders.get(original['id'])['text'], '削笔')
        updated = agent_store.confirm(self.sid, draft['id'], draft['revision'])
        self.assertEqual(updated['id'], original['id'])
        self.assertEqual(updated['text'], '喝水')

    def test_button_confirmation_is_idempotent_and_clear_rotates_session(self):
        row = self.draft()
        client = TestClient(app, headers={'X-Robot-Client': 'test'})
        data = {'session_id': 'browser-a', 'revision': row['revision']}
        url = '/api/reminder-drafts/' + row['id'] + '/confirm'
        self.assertEqual(client.post(url, json=data).status_code, 200)
        self.assertEqual(client.post(url, json=data).status_code, 200)
        self.assertEqual(len(reminders.list_active()), 1)
        self.assertEqual(client.delete('/api/conversations/browser-a').status_code, 200)
        self.assertNotEqual(agent_store.session('browser-a')['harness_id'], self.sid)
        self.assertEqual(client.post(url, json=data).status_code, 409)

    def test_all_chat_routes_use_harness_without_rule_interception(self):
        client = TestClient(app, headers={'X-Robot-Client': 'test'})
        async def run(sid, text, on_event=lambda event: None):
            return {'reply': '收到：' + text, 'drafts': [], 'source': 'harness'}
        with patch.object(settings, 'agent_backend', 'harness'), patch.object(agent, 'run', side_effect=run) as mock:
            for route in ['/api/chat', '/api/command', '/api/command/stream']:
                response = client.post(route, params={'text': '提醒我明天晚上八点练琴', 'session_id': 'route-test'})
                self.assertEqual(response.status_code, 200)
                self.assertIn('收到', response.text)
            self.assertEqual(mock.call_count, 3)


class AgentCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_closing_stream_stops_running_request(self):
        service = HarnessService()
        stopped = asyncio.Event()
        async def run(*args):
            try:
                await asyncio.sleep(60)
            finally:
                stopped.set()
        with patch.object(service, 'run', side_effect=run):
            stream = service.stream('a', '天气怎么样')
            await anext(stream)
            await asyncio.sleep(0)
            await stream.aclose()
            self.assertTrue(stopped.is_set())

    async def test_failed_stream_has_no_success_event(self):
        service = HarnessService()
        with patch.object(service, 'run', side_effect=RuntimeError('测试错误')):
            events = [e async for e in service.stream('a', '你好')]
        self.assertEqual(events[-1]['type'], 'error')
        self.assertNotIn('done', [e['type'] for e in events])
