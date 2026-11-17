"""纯语音对话引擎单测：用 sherpa-onnx 自带测试词+测试 wav 驱动状态机，不依赖真实麦克风。

模型/依赖缺失时整体跳过（与 vision 优雅降级一致）。
"""
import unittest
import wave
from pathlib import Path

import numpy as np

from app import realtime

MD = realtime.model_dir()
KWS = realtime.kws_dir()
HAVE = realtime.RealtimeVoiceService().available()


def _readwav(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def _test_spotter():
    import sherpa_onnx
    return sherpa_onnx.KeywordSpotter(
        tokens=str(KWS / "tokens.txt"),
        encoder=str(KWS / "encoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
        decoder=str(KWS / "decoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
        joiner=str(KWS / "joiner-epoch-99-avg-1-chunk-16-left-64.onnx"),
        keywords_file=str(KWS / "test_wavs" / "test_keywords.txt"),
        keywords_threshold=0.25, keywords_score=1.0, num_threads=2)


def _test_vad():
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(MD / "silero_vad.onnx")
    cfg.silero_vad.threshold = 0.5
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.min_speech_duration = 0.25
    cfg.silero_vad.window_size = 512
    cfg.sample_rate = 16000
    cfg.num_threads = 1
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)


@unittest.skipUnless(HAVE, "需要 sherpa-onnx 与 KWS/VAD 模型")
class RealtimeEngineTests(unittest.TestCase):
    def _feed(self, sess, samples):
        events = []
        for i in range(0, len(samples), 1600):
            events.extend(sess.accept(samples[i:i + 1600]))
        return events

    def test_wake_then_endpoint(self):
        sess = realtime.VoiceSession(_test_spotter(), _test_vad(), wake_label="文森特卡索")
        s = _readwav(KWS / "test_wavs" / "3.wav")
        s = np.concatenate([s, np.zeros(int(1.5 * 16000), dtype=np.float32)])
        events = self._feed(sess, s)
        types = [e["type"] for e in events]
        self.assertIn("wake", types)
        self.assertIn("endpoint", types)
        ep = next(e for e in events if e["type"] == "endpoint")
        self.assertGreater(len(ep["pcm"]) / 16000, 0.6)
        self.assertEqual(sess.state, "processing")

    def test_processing_ignores_mic(self):
        sess = realtime.VoiceSession(_test_spotter(), _test_vad(), wake_label="文森特卡索")
        sess.set_state("processing")
        self.assertEqual(sess.accept(np.ones(1600, dtype=np.float32)), [])

    def test_barge_in_during_speaking(self):
        # 全双工：说话期间喊唤醒词 → barge_in 打断，转 capturing
        sess = realtime.VoiceSession(_test_spotter(), _test_vad(), wake_label="文森特卡索")
        sess.set_state("speaking")
        s = _readwav(KWS / "test_wavs" / "3.wav")
        s = np.concatenate([s, np.zeros(int(1.5 * 16000), dtype=np.float32)])
        events = self._feed(sess, s)
        types = [e["type"] for e in events]
        self.assertIn("barge_in", types)
        self.assertNotIn("wake", types)
        self.assertIn("endpoint", types)

    def test_speaking_ignores_non_wake_audio(self):
        # 说话期间非唤醒音频（如回声/噪声）不打断
        sess = realtime.VoiceSession(_test_spotter(), _test_vad(), wake_label="文森特卡索")
        sess.set_state("speaking")
        events = self._feed(sess, np.zeros(int(1 * 16000), dtype=np.float32))
        self.assertEqual(events, [])
        self.assertEqual(sess.state, "speaking")

    def test_back_to_listening_resets(self):
        sess = realtime.VoiceSession(_test_spotter(), _test_vad(), wake_label="文森特卡索")
        sess.set_state("processing")
        sess.set_state("listening")
        self.assertEqual(sess.state, "listening")

    def test_silence_no_wake(self):
        sess = realtime.VoiceSession(_test_spotter(), _test_vad(), wake_label="文森特卡索")
        events = self._feed(sess, np.zeros(int(2 * 16000), dtype=np.float32))
        self.assertEqual(events, [])
        self.assertEqual(sess.state, "listening")


class RealtimeGracefulTests(unittest.TestCase):
    def test_wake_keyword_pinyin_config(self):
        # 唤醒词拼音可被模型 token 覆盖（ā/b/àn 均在 tokens.txt）
        tokens = (KWS / "tokens.txt").read_text(encoding="utf-8") if (KWS / "tokens.txt").is_file() else ""
        if tokens:
            for tok in realtime.WAKE_PINYIN.split():
                self.assertIn(tok + " ", tokens)


if __name__ == "__main__":
    unittest.main()
