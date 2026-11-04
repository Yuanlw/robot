import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

_test_data = tempfile.TemporaryDirectory(prefix='robot-tests-')
os.environ['ROBOT_DATA_DIR'] = _test_data.name
os.environ['ROBOT_LLM_API_KEY'] = ''
from fastapi.testclient import TestClient
from app.main import app, llm, voice, conversation_histories
from app.llm import LLMService
from app.config import settings
from app.voice import VoiceService, spoken_text
from app.db import init_db, get_conversation_memory


def sse(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]


class ConversationTests(unittest.TestCase):
    def setUp(self):
        backend = patch.object(settings, 'agent_backend', 'legacy')
        backend.start()
        self.addCleanup(backend.stop)
        init_db()
        from app.agent_store import init_agent_db
        init_agent_db()
        self.client = TestClient(app)

    def test_local_reminder_does_not_call_model(self):
        with patch.object(llm, 'stream_chat', side_effect=AssertionError('local reminder must not call model')):
            events = sse(self.client.post('/api/command/stream', params={'text': '提醒我明天晚上八点练琴'}))
        self.assertEqual(events[1]['type'], 'result')
        self.assertTrue(events[1]['needs_confirmation'])
        self.assertEqual(events[-1]['type'], 'done')

    def test_stream_yields_text_and_completion(self):
        with patch.object(llm, 'stream_messages', return_value=iter(['你好，', '家人们。'])):
            events = sse(self.client.post('/api/command/stream', params={'text': '你好'}))
        self.assertEqual(''.join(x['text'] for x in events if x['type']=='delta'), '你好，家人们。')
        self.assertEqual(events[-1]['type'], 'done')

    def test_conversation_context_is_retained_and_clearable(self):
        conversation_histories.pop('context-test', None)
        with patch.object(llm, 'stream_messages', side_effect=lambda _: iter(['记住了。'])) as model:
            first = sse(self.client.post('/api/command/stream', params={'text': '我叫小明', 'session_id': 'context-test'}))
            second = sse(self.client.post('/api/command/stream', params={'text': '我叫什么？', 'session_id': 'context-test'}))
        self.assertEqual(first[-1]['type'], 'done')
        sent_messages = model.call_args.args[0]
        self.assertEqual(sent_messages[-3]['content'], '我叫小明')
        self.assertEqual(sent_messages[-1]['content'], '我叫什么？')
        self.assertEqual(len(conversation_histories['context-test']['messages']), 4)
        conversation_histories.pop('context-test')
        self.assertEqual(len(get_conversation_memory('context-test')['messages']), 4)
        self.assertEqual(self.client.delete('/api/conversations/context-test').status_code, 200)
        self.assertNotIn('context-test', conversation_histories)

    def test_old_turns_are_summarized_before_model_request(self):
        conversation_histories['long-test'] = {
            'summary': '',
            'messages': [{'role': role, 'content': f'内容{i}'} for i in range(20) for role in ('user', 'assistant')][:20],
        }
        with patch.object(llm, 'summarize', return_value='用户喜欢被称为小明。') as compact, \
             patch.object(llm, 'stream_messages', side_effect=lambda messages: iter(['收到。'])) as model:
            sse(self.client.post('/api/command/stream', params={'text': '继续', 'session_id': 'long-test'}))
        compact.assert_called_once()
        sent = model.call_args.args[0]
        self.assertIn('用户喜欢被称为小明。', sent[0]['content'])
        self.assertEqual(sent[-1]['content'], '继续')

    def test_partial_failure_is_not_success(self):
        def broken(_):
            yield '部分文字'
            raise RuntimeError('聊天模型响应超时')
        with patch.object(llm, 'stream_messages', side_effect=broken):
            events = sse(self.client.post('/api/command/stream', params={'text': '你好'}))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertNotIn('done', [x['type'] for x in events])

    def test_invalid_time_returns_client_error(self):
        r = self.client.post('/api/command/stream', params={'text':'提醒我明天25点练琴'})
        self.assertEqual(r.status_code,400)

    def test_health_does_not_claim_verified_connection(self):
        self.assertEqual(self.client.get('/api/health').json()['llm_status'], 'not_configured')

    def test_audio_endpoint_and_validation(self):
        with patch.object(voice, 'synthesize', return_value=b'RIFFtest-wave'):
            r=self.client.post('/api/voice/audio',json={'text':'你好'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers['content-type'],'audio/wav')
        self.assertEqual(self.client.post('/api/voice/audio',json={'text':''}).status_code,422)
        self.assertEqual(self.client.post('/api/voice/audio',json={'text':'好'*1501}).status_code,422)

    def test_voice_failure_is_visible(self):
        with patch.object(voice,'synthesize',side_effect=RuntimeError('test')):
            self.assertEqual(self.client.post('/api/voice/audio',json={'text':'你好'}).status_code,503)

    def test_voice_choices_and_persistent_selection(self):
        with patch.object(voice, 'neural_available', return_value=True), patch.object(voice, 'argmax_available', return_value=True):
            choices = self.client.get('/api/voice/settings').json()
            self.assertIn(choices['voice'], [v['id'] for v in choices['available_voices']])
            self.assertEqual(len(choices['available_voices']), 6)
            saved = self.client.put('/api/voice/settings', json={'voice': 'argmax:serena'})
            self.assertEqual(saved.status_code, 200)
            self.assertEqual(saved.json()['voice'], 'argmax:serena')
            self.assertEqual(self.client.get('/api/voice/settings').json()['voice'], 'argmax:serena')
            invalid = self.client.put('/api/voice/settings', json={'voice': 'unknown'})
            self.assertEqual(invalid.status_code, 422)

    def test_reminder_create_edit_and_cancel_flow(self):
        init_db()
        due = (datetime.now().astimezone() + timedelta(days=2)).isoformat()
        created = self.client.post('/api/reminders', json={'text': '练琴', 'due_at': due})
        self.assertEqual(created.status_code, 200)
        reminder_id = created.json()['id']
        changed_due = (datetime.now().astimezone() + timedelta(days=3)).isoformat()
        edited = self.client.patch(f'/api/reminders/{reminder_id}', json={'text': '复习英语', 'due_at': changed_due})
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()['text'], '复习英语')
        self.assertEqual(self.client.delete(f'/api/reminders/{reminder_id}').status_code, 200)
        self.assertNotIn(reminder_id, [item['id'] for item in self.client.get('/api/reminders').json()])

    def test_reminder_rejects_past_time_and_blank_content(self):
        past = (datetime.now().astimezone() - timedelta(minutes=1)).isoformat()
        self.assertEqual(self.client.post('/api/reminders', json={'text': '过期', 'due_at': past}).status_code, 422)
        future = (datetime.now().astimezone() + timedelta(days=1)).isoformat()
        self.assertEqual(self.client.post('/api/reminders', json={'text': '   ', 'due_at': future}).status_code, 422)

    def test_request_uses_short_low_reasoning_stream(self):
        service=LLMService()
        with patch.object(settings,'llm_reasoning_effort','low'), patch.object(settings,'llm_thinking',None):
            request=service._request('你好')
        self.assertTrue(request['stream'])
        self.assertEqual(request['extra_body']['reasoning_effort'],'low')
        self.assertEqual(request['max_tokens'],160)
        with patch.object(settings,'llm_reasoning_effort',''),patch.object(settings,'llm_thinking',None):
            self.assertNotIn('extra_body',service._request('你好'))

    def test_bailian_explicitly_disables_thinking(self):
        with patch.object(settings,'llm_thinking','disabled'),patch.object(settings,'llm_reasoning_effort','low'):
            request=LLMService()._request('你好')
        self.assertEqual(request['extra_body'],{'enable_thinking': False})

    def test_model_timeout_and_auth_errors(self):
        import httpx
        from openai import APITimeoutError, PermissionDeniedError
        for error in [APITimeoutError(request=httpx.Request('POST','https://example.test')),PermissionDeniedError('no access',response=httpx.Response(403,request=httpx.Request('POST','https://example.test')),body=None)]:
            service=LLMService()
            service._client=MagicMock()
            service._client.chat.completions.create.side_effect=error
            with self.assertRaises(RuntimeError):
                list(service.stream_chat('你好'))
            self.assertIn(service.last_status, ['timeout','error'])

    def test_mandarin_preprocessing(self):
        self.assertEqual(spoken_text('**提醒**：[练琴](https://example.test)，20:30开始。'), '提醒：练琴，20点30分开始。')

    def test_neural_audio_cache(self):
        v=VoiceService()
        with patch.object(v,'neural_available',return_value=True),patch.object(v,'_neural',return_value=b'RIFFaudio') as model:
            self.assertEqual(v.synthesize('你好'),b'RIFFaudio')
            self.assertEqual(v.synthesize('你好'),b'RIFFaudio')
            self.assertEqual(model.call_count,1)

    def test_neural_failure_falls_back_explicitly(self):
        v=VoiceService()
        with patch.object(v,'neural_available',return_value=True),patch.object(v,'_neural',side_effect=RuntimeError('fail')),patch.object(v,'_system',return_value=b'RIFFfallback'):
            self.assertEqual(v.synthesize('你好'),b'RIFFfallback')
            self.assertIsNotNone(v.details()['fallback_reason'])

if __name__=='__main__':
    unittest.main()
