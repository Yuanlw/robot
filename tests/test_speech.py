"""统一语音通道（Speech Policy Engine）单元测试。

不依赖真实音频：speak_fn 用替身，osascript 静音检测打桩。
"""
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from app import speech
from app.speech import SpeechChannel, PRIORITY_REMINDER, PRIORITY_RESPONSE, PRIORITY_STUDY, PRIORITY_POSTURE


def make_hour(h):
    return datetime(2026, 11, 13, h, 0).astimezone()


class ChannelCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='robot-speech-')
        self.addCleanup(self.tmp.cleanup)
        self.dnd_file = Path(self.tmp.name) / 'dnd.json'
        self.spoken = []
        self.speak_result = 'spoken'

        def speak_fn(text):
            self.spoken.append(text)
            return self.speak_result

        self.ch = SpeechChannel(speak_fn, dnd_file=self.dnd_file, quiet_start=22, quiet_end=7)
        # 默认打桩：非静音、系统检测关闭（各测试自行覆盖）
        self.ch.system_muted = lambda: False

    def drain(self, n=10):
        for _ in range(n):
            if not self.ch._process_next(block=False):
                break


class QuietHoursTests(ChannelCase):
    def test_wrap_around_midnight(self):
        for h, expect in [(22, True), (23, True), (0, True), (6, True), (7, False), (12, False), (21, False)]:
            self.assertEqual(self.ch.in_quiet(make_hour(h)), expect, f"hour={h}")

    def test_quiet_suppresses_autonomous_but_not_reminder_or_response(self):
        self.ch.in_quiet = lambda now=None: True
        self.assertEqual(self.ch.submit('该休息了', PRIORITY_STUDY), 'suppress_quiet')
        self.assertEqual(self.ch.submit('坐姿提醒', PRIORITY_POSTURE), 'suppress_quiet')
        self.assertEqual(self.spoken, [])
        # P0 显式提醒、P1 交互应答不受安静时段限制
        self.assertEqual(self.ch.submit('提醒到点', PRIORITY_REMINDER), 'queued')
        self.assertEqual(self.ch.submit('试听', PRIORITY_RESPONSE), 'queued')


class DndTests(ChannelCase):
    def test_dnd_file_roundtrip(self):
        until = speech.set_dnd(60, path=self.dnd_file)
        self.assertTrue(speech.dnd_until(path=self.dnd_file) is not None)
        speech.clear_dnd(path=self.dnd_file)
        self.assertIsNone(speech.dnd_until(path=self.dnd_file))

    def test_dnd_expired_file_ignored(self):
        self.dnd_file.write_text('{"until": "2020-01-01T00:00:00+08:00"}')
        self.assertFalse(self.ch.dnd_active())

    def test_dnd_semantics_by_priority(self):
        speech.set_dnd(60, path=self.dnd_file)
        self.assertEqual(self.ch.submit('该休息了', PRIORITY_STUDY), 'suppress_dnd')   # 自主播报压制
        self.assertEqual(self.ch.submit('提醒到点', PRIORITY_REMINDER), 'deferred_dnd')  # 提醒延后不入队
        self.assertEqual(self.ch.submit('试听', PRIORITY_RESPONSE, wait=False), 'queued')  # 用户动作回应放行
        self.drain()
        self.assertEqual(self.spoken, ['试听'])

    def test_minutes_clamped(self):
        speech.set_dnd(0, path=self.dnd_file)
        self.assertTrue(speech.dnd_until(path=self.dnd_file) is not None)  # 下限 1 分钟
        speech.set_dnd(99999, path=self.dnd_file)
        until = speech.dnd_until(path=self.dnd_file)
        self.assertLess((until - datetime.now().astimezone()).total_seconds(), 8 * 3600 + 60)


class MutedTests(ChannelCase):
    def test_muted_reminder_not_delivered(self):
        self.ch.system_muted = lambda: True
        self.assertEqual(self.ch.submit('吃药', PRIORITY_REMINDER), 'muted')
        self.assertEqual(self.spoken, [])  # 静音时根本不播，也就不会假送达

    def test_muted_stops_autonomous(self):
        self.ch.system_muted = lambda: True
        self.assertEqual(self.ch.submit('该休息了', PRIORITY_STUDY), 'muted')

    def test_muted_during_queue_blocks_reminder_at_dequeue(self):
        # 入队后才静音：出队时二次检查，提醒不播、回执 muted
        item_result = []
        self.ch.submit('吃药', PRIORITY_REMINDER, wait=False)
        self.ch.system_muted = lambda: True
        self.drain()
        self.assertEqual(self.spoken, [])


class PriorityAndFreshnessTests(ChannelCase):
    def test_priority_order(self):
        # worker 未启动时连续入队，出队顺序 = 优先级序（低数字先播）
        self.ch.submit('柔性确认', PRIORITY_POSTURE + 1)
        self.ch.submit('坐姿', PRIORITY_POSTURE)
        self.ch.submit('休息', PRIORITY_STUDY)
        self.drain()
        self.assertEqual(self.spoken, ['休息', '坐姿', '柔性确认'])

    def test_stale_dropped_not_spoken(self):
        self.ch.submit('过期的话', PRIORITY_STUDY, fresh_seconds=-1)
        self.drain()
        self.assertEqual(self.spoken, [])
        st = self.ch.status()
        self.assertTrue(any(s['reason'] == '已过期' for s in st['suppressed']))

    def test_fresh_within_window_spoken(self):
        self.ch.submit('新鲜的话', PRIORITY_STUDY, fresh_seconds=120)
        self.drain()
        self.assertEqual(self.spoken, ['新鲜的话'])

    def test_failed_speak_returns_failed(self):
        # afplay 不存在/失败（返回 text-only）：尝试过、结果如实记为 failed（不算送达）
        events = []
        self.ch._on_event = lambda ev: events.append(ev)
        self.speak_result = 'text-only'
        self.ch.submit('说不了', PRIORITY_STUDY)
        self.drain()
        self.assertEqual(self.spoken, ['说不了'])
        self.assertIn({'type': 'speech_result', 'status': 'failed', 'text': '说不了'}, events)


class WorkerTests(ChannelCase):
    def test_wait_delivered(self):
        self.ch.start()
        self.addCleanup(self.ch.stop)
        self.assertEqual(self.ch.submit('提醒到点', PRIORITY_REMINDER, wait=True, timeout=5), 'delivered')
        self.assertEqual(self.spoken, ['提醒到点'])

    def test_speak_exception_returns_failed(self):
        def boom(text):
            raise RuntimeError('afplay died')
        self.ch._speak = boom
        self.ch.start()
        self.addCleanup(self.ch.stop)
        self.assertEqual(self.ch.submit('提醒到点', PRIORITY_REMINDER, wait=True, timeout=5), 'failed')

    def test_suppressed_dedup(self):
        self.ch.in_quiet = lambda now=None: True
        self.ch.submit('同一句', PRIORITY_STUDY)
        self.ch.submit('同一句', PRIORITY_STUDY)
        st = self.ch.status()
        self.assertEqual(len(st['suppressed']), 1)  # 同句同因不重复刷屏


class StatusTests(ChannelCase):
    def test_status_shape(self):
        st = self.ch.status()
        for key in ('dnd_active', 'dnd_until', 'quiet_now', 'quiet_hours', 'muted', 'queue_len', 'suppressed'):
            self.assertIn(key, st)
        self.assertEqual(st['quiet_hours'], '22:00-07:00')


if __name__ == '__main__':
    unittest.main()
