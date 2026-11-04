import asyncio
import base64
import io
import json
import math
import os
import struct
import tempfile
import unittest
import uuid
import wave
from unittest.mock import patch

import httpx

_test_data = tempfile.TemporaryDirectory(prefix='robot-bailian-tests-')
os.environ.setdefault('ROBOT_DATA_DIR', _test_data.name)
# 本模块按字母序最先导入、settings 单例在此构建：必须在导入 app 前清掉密钥，
# 防止读到本机 .env 的真实 key 后测试误调云端（其他测试模块沿用空 key 语义）。
os.environ['ROBOT_LLM_API_KEY'] = ''

from fastapi.testclient import TestClient
from app import bailian
from app.bailian import BailianError, audio_stats, is_silence, pcm_to_wav, stt_transcribe, tts_synthesize
from app.config import settings
from app.db import init_db
from app.agent_store import init_agent_db
from app.harness import agent
from app.main import app
from app.voice import VoiceService

BASE = 'https://example.com/compatible-mode/v1'


def tone_pcm(seconds=0.5, rate=16000, freq=440, amp=3000):
    n = int(seconds * rate)
    return struct.pack(f'<{n}h', *[int(amp * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)])


def silence_pcm(seconds=0.5, rate=16000):
    return b'\x00\x00' * int(seconds * rate)


def wav_bytes(pcm, rate=16000):
    return pcm_to_wav(pcm, rate)


class PcmUtilsTests(unittest.TestCase):
    def test_pcm_to_wav_header(self):
        data = pcm_to_wav(tone_pcm(0.2), 16000)
        with wave.open(io.BytesIO(data)) as f:
            self.assertEqual(f.getframerate(), 16000)
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 2)
            self.assertEqual(f.getnframes(), 3200)

    def test_silence_gate(self):
        silent, stats = is_silence(silence_pcm(1.0))
        self.assertTrue(silent)
        self.assertEqual(stats['peak'], 0)
        voiced, stats2 = is_silence(tone_pcm(1.0))
        self.assertFalse(voiced)
        self.assertGreater(stats2['peak'], 64)

    def test_empty_audio_is_silent(self):
        self.assertTrue(is_silence(b'')[0])


class SttTranscribeTests(unittest.TestCase):
    def _run(self, handler, **kwargs):
        transport = httpx.MockTransport(handler)
        return stt_transcribe(tone_pcm(0.3), api_key='k', base_url=BASE,
                              model='qwen3-asr-flash', transport=transport, **kwargs)

    def test_openai_shape(self):
        seen = {}

        def handler(request):
            seen['url'] = str(request.url)
            seen['body'] = json.loads(request.content)
            return httpx.Response(200, json={'choices': [{'message': {'content': ' 今天天气怎么样 '}}]})

        text = self._run(handler)
        self.assertEqual(text, '今天天气怎么样')
        self.assertTrue(seen['url'].endswith('/compatible-mode/v1/chat/completions'))
        self.assertEqual(seen['body']['model'], 'qwen3-asr-flash')
        self.assertFalse(seen['body']['stream'])
        audio = seen['body']['messages'][-1]['content'][0]['input_audio']['data']
        self.assertTrue(audio.startswith('data:audio/wav;base64,'))

    def test_empty_content_is_legal(self):
        text = self._run(lambda request: httpx.Response(200, json={'choices': [{'message': {'content': ''}}]}))
        self.assertEqual(text, '')

    def test_native_shape_fallback(self):
        text = self._run(lambda request: httpx.Response(200, json={'output': {'sentence': {'text': '你好'}}}))
        self.assertEqual(text, '你好')

    def test_structural_error_raises(self):
        with self.assertRaises(BailianError):
            self._run(lambda request: httpx.Response(200, json={'code': 'InternalError', 'message': 'boom'}))

    def test_http_error_hint(self):
        with self.assertRaises(BailianError) as ctx:
            self._run(lambda request: httpx.Response(401, text='invalid key'))
        self.assertIn('鉴权失败', str(ctx.exception))

    def test_requires_key(self):
        with patch.object(settings, 'llm_api_key', None):
            with self.assertRaises(BailianError):
                stt_transcribe(tone_pcm(0.2))

    def test_empty_pcm_rejected(self):
        with self.assertRaises(BailianError):
            stt_transcribe(b'', api_key='k', base_url=BASE, transport=httpx.MockTransport(lambda r: None))


class TtsSynthesizeTests(unittest.TestCase):
    @staticmethod
    def _sse(chunks, rate=24000):
        body = b''
        for c in chunks:
            payload = {'output': {'audio': {'data': base64.b64encode(c).decode()}}}
            body += b'id:1\nevent:result\n:' + str(len(body)).encode() + b'\ndata:' + json.dumps(payload).encode() + b'\n\n'
        body += b'id:99\nevent:task-finished\ndata:{"output":{}}\n\n'
        return body

    def test_stream_to_wav(self):
        pcm1 = tone_pcm(0.1, 24000)
        pcm2 = tone_pcm(0.1, 24000, freq=660)
        seen = {}

        def handler(request):
            seen['url'] = str(request.url)
            seen['sse'] = request.headers.get('X-DashScope-SSE')
            seen['body'] = json.loads(request.content)
            return httpx.Response(200, content=self._sse([pcm1, pcm2]))

        wav_data, first_ms = tts_synthesize('你好世界', api_key='k', base_url=BASE,
                                            model='qwen3-tts-flash', voice='Cherry',
                                            transport=httpx.MockTransport(handler))
        self.assertEqual(seen['sse'], 'enable')
        self.assertTrue(seen['url'].endswith('/api/v1/services/aigc/multimodal-generation/generation'))
        self.assertFalse('/compatible-mode' in seen['url'])
        self.assertEqual(seen['body']['input'], {'text': '你好世界', 'voice': 'Cherry', 'language_type': 'Chinese'})
        self.assertIsNotNone(first_ms)
        with wave.open(io.BytesIO(wav_data)) as f:
            self.assertEqual(f.getframerate(), 24000)
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getnframes(), (len(pcm1) + len(pcm2)) // 2)

    def test_retry_on_500_then_success(self):
        calls = {'n': 0}

        def handler(request):
            calls['n'] += 1
            if calls['n'] == 1:
                return httpx.Response(500, text='boom')
            return httpx.Response(200, content=self._sse([tone_pcm(0.05, 24000)]))

        wav_data, _ = tts_synthesize('重试', api_key='k', base_url=BASE,
                                     transport=httpx.MockTransport(handler))
        self.assertEqual(calls['n'], 2)
        self.assertTrue(wav_data.startswith(b'RIFF'))

    def test_no_retry_on_401(self):
        calls = {'n': 0}

        def handler(request):
            calls['n'] += 1
            return httpx.Response(401, text='bad key')

        with self.assertRaises(BailianError):
            tts_synthesize('你好', api_key='k', base_url=BASE, transport=httpx.MockTransport(handler))
        self.assertEqual(calls['n'], 1)

    def test_business_code_raises(self):
        body = b'data:' + json.dumps({'code': 'InvalidParameter', 'message': 'voice required'}).encode() + b'\n\n'
        with self.assertRaises(BailianError) as ctx:
            tts_synthesize('你好', api_key='k', base_url=BASE,
                           transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body)))
        self.assertIn('InvalidParameter', str(ctx.exception))

    def test_long_text_truncated(self):
        seen = {}

        def handler(request):
            seen['body'] = json.loads(request.content)
            return httpx.Response(200, content=self._sse([tone_pcm(0.05, 24000)]))

        tts_synthesize('字' * 900, api_key='k', base_url=BASE, transport=httpx.MockTransport(handler))
        self.assertEqual(len(seen['body']['input']['text']), 600)


class VoiceServiceBailianTests(unittest.TestCase):
    def setUp(self):
        self.voice = VoiceService()
        backend = patch.object(settings, 'tts_voice', 'bailian:Cherry')
        key = patch.object(settings, 'llm_api_key', 'test-key')
        backend.start(); key.start()
        self.addCleanup(backend.stop); self.addCleanup(key.stop)

    def test_synthesize_uses_bailian(self):
        with patch.object(bailian, 'tts_synthesize', return_value=(b'RIFFfake', 321.0)) as mock_tts:
            audio = self.voice.synthesize('你好。')
        self.assertEqual(audio, b'RIFFfake')
        self.assertEqual(self.voice.last_bailian_first_chunk_ms, 321.0)
        self.assertEqual(mock_tts.call_args.kwargs['voice'], 'Cherry')
        self.assertEqual(self.voice.status(), 'bailian-zh-cloud')

    def test_bailian_failure_falls_back_local(self):
        with patch.object(bailian, 'tts_synthesize', side_effect=BailianError('超时')), \
                patch.object(self.voice, '_system', return_value=b'syswav'):
            audio = self.voice.synthesize('你好。')
        self.assertEqual(audio, b'syswav')
        self.assertIn('回退', self.voice._bailian_error)
        self.assertNotEqual(self.voice.status(), 'bailian-zh-cloud')
        self.assertIn('百炼', self.voice.details()['fallback_reason'])

    def test_details_include_bailian_choice(self):
        details = self.voice.details()
        ids = [item['id'] for item in details['available_voices']]
        self.assertIn('bailian:Cherry', ids)
        self.assertEqual(details['voice'], 'bailian:Cherry')

    def test_set_voice_requires_key(self):
        service = VoiceService()
        with patch.object(settings, 'llm_api_key', None):
            with self.assertRaises(RuntimeError):
                service.set_voice('bailian:Cherry')


class SttEndpointTests(unittest.TestCase):
    def setUp(self):
        init_db()
        self.client = TestClient(app)

    def test_worklet_js_served(self):
        response = self.client.get('/static/worklet.js')
        self.assertEqual(response.status_code, 200)
        self.assertIn('pcm-capture', response.text)
        self.assertIn('application/javascript', response.headers['content-type'])

    def test_browser_backend_returns_501(self):
        with patch.object(settings, 'stt_backend', 'browser'):
            response = self.client.post('/api/stt', files={'audio': ('s.wav', wav_bytes(tone_pcm(0.2)), 'audio/wav')})
        self.assertEqual(response.status_code, 501)

    def test_silent_upload_short_circuits(self):
        with patch.object(bailian, 'stt_transcribe') as mock_stt:
            response = self.client.post('/api/stt', files={'audio': ('s.wav', wav_bytes(silence_pcm(1.0)), 'audio/wav')})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['silent'])
        self.assertEqual(body['text'], '')
        mock_stt.assert_not_called()

    def test_speech_upload_transcribed(self):
        with patch.object(bailian, 'stt_transcribe', return_value='明天提醒我买菜') as mock_stt:
            response = self.client.post('/api/stt', files={'audio': ('s.wav', wav_bytes(tone_pcm(0.4)), 'audio/wav')})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['text'], '明天提醒我买菜')
        self.assertFalse(body['silent'])
        self.assertIn('stt_ms', body)
        mock_stt.assert_called_once()

    def test_bad_audio_rejected(self):
        response = self.client.post('/api/stt', files={'audio': ('s.wav', b'not-a-wav', 'audio/wav')})
        self.assertEqual(response.status_code, 422)

    def test_provider_error_maps_502(self):
        with patch.object(bailian, 'stt_transcribe', side_effect=BailianError('限流')):
            response = self.client.post('/api/stt', files={'audio': ('s.wav', wav_bytes(tone_pcm(0.4)), 'audio/wav')})
        self.assertEqual(response.status_code, 502)


class FakeProc:
    returncode = None


class FakeSDK:
    instances = []

    def __init__(self, context_path, home, on_event):
        self.context_path, self.home, self.on_event = context_path, home, on_event
        self.answers, self.calls, self.closed = [], [], False
        self.proc = FakeProc()
        FakeSDK.instances.append(self)

    async def start(self):
        self.calls.append('initialize')

    async def rpc(self, method, params=None):
        self.calls.append(method)
        if method == 'session/prompt':
            self.answers.append('好的')
        if method == 'session/wait' and settings.harness_max_tokens == -1:
            raise RuntimeError('simulated failure')
        return {}

    async def close(self):
        self.closed = True
        self.proc.returncode = 0


class WarmPoolTests(unittest.TestCase):
    def setUp(self):
        init_db()
        init_agent_db()
        agent.warm.clear()
        agent.locks.clear()
        agent.reaper = None
        agent.warm_guard = asyncio.Lock()
        FakeSDK.instances = []
        sdk = patch('app.harness.SDK', FakeSDK)
        key = patch.object(settings, 'llm_api_key', 'test-key')
        sdk.start(); key.start()
        self.addCleanup(sdk.stop); self.addCleanup(key.stop)

    def test_two_turns_reuse_one_process(self):
        browser = 'warm-' + uuid.uuid4().hex[:8]

        async def scenario():
            first = await agent.run(browser, '你好')
            second = await agent.run(browser, '再见')
            await agent.shutdown()
            return first, second

        first, second = asyncio.run(scenario())
        self.assertEqual(first['reply'], '好的')
        self.assertEqual(second['reply'], '好的')
        self.assertEqual(len(FakeSDK.instances), 1)
        client = FakeSDK.instances[0]
        self.assertEqual(client.calls.count('initialize'), 1)
        self.assertEqual(client.calls.count('session/prompt'), 2)
        # 第二轮：会话已在进程内，无需再 resume
        self.assertEqual(client.calls.count('session/resume'), 0)
        self.assertTrue(client.closed)
        self.assertEqual(agent.warm, {})
        # 固定 context 路径：常驻 MCP 每轮重读同一文件
        self.assertEqual(client.context_path.name, 'agent-context.json')

    def test_context_file_refreshed_each_turn(self):
        browser = 'warm-' + uuid.uuid4().hex[:8]

        async def scenario():
            await agent.run(browser, '第一轮')
            snapshot = json.loads(FakeSDK.instances[0].context_path.read_text())
            await agent.run(browser, '第二轮')
            snapshot2 = json.loads(FakeSDK.instances[0].context_path.read_text())
            await agent.shutdown()
            return snapshot, snapshot2

        first_ctx, second_ctx = asyncio.run(scenario())
        self.assertEqual(first_ctx['session_id'], second_ctx['session_id'])
        self.assertNotEqual(first_ctx['received_at'], second_ctx['received_at'])

    def test_failed_turn_kills_process_and_respawns(self):
        browser = 'warm-' + uuid.uuid4().hex[:8]

        async def scenario():
            await agent.run(browser, '你好')
            tokens = patch.object(settings, 'harness_max_tokens', -1)
            tokens.start()
            try:
                with self.assertRaises(RuntimeError):
                    await agent.run(browser, '会失败的轮次')
            finally:
                tokens.stop()
            await agent.run(browser, '恢复后再来')
            await agent.shutdown()

        asyncio.run(scenario())
        self.assertEqual(len(FakeSDK.instances), 2)
        self.assertTrue(FakeSDK.instances[0].closed)
        self.assertTrue(FakeSDK.instances[1].closed)


if __name__ == '__main__':
    unittest.main()
