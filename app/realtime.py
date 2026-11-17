"""学习模式纯语音对话引擎（阶段 B）：常开麦 + 唤醒词 + VAD 自动断句 + barge-in 打断。

设计（全双工打断，免按钮）：
- listening：麦克风常开，音频喂 KWS 等唤醒词"阿伴阿伴"；
- capturing：听到唤醒词后收录这句（含唤醒前 ~1s 预缓冲），VAD 检测说完（静音）→ endpoint；
- speaking：播放回答期间仍听唤醒词——用户喊"阿伴阿伴"即 barge_in 打断，转 capturing；
  回声由浏览器 AEC 消除（机器人回答 rarely 含唤醒词，自唤醒风险低）；
- processing：识别/思考中忽略麦克风，由 WebSocket 处理层置位。

隐私红线：音频只在内存流过本地 sherpa-onnx 模型，不录、不存、不上传；
唤醒后的语音按现有 /api/stt 隐私边界送百炼识别。

纯状态机 + 本地模型，可用合成/自带测试音频单测，不依赖真实麦克风。
sherpa-onnx 或模型缺失时 available()=False，功能整体优雅降级（回退按钮模式）。
"""
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import settings

SAMPLE_RATE = 16000
WAKE_LABEL = "阿伴阿伴"
WAKE_PINYIN = "ā b àn ā b àn"
PREBUFFER_SECONDS = 1.0      # 唤醒前预缓冲，避免削掉问句开头
MAX_CAPTURE_SECONDS = 15.0   # 单句最长收录，防失控
MIN_UTTERANCE_SECONDS = 0.6  # 短于此的段视为仅唤醒词/噪声，丢弃继续听
VAD_MIN_SILENCE = 0.5        # 说完判定：静音超过此时长 → 断句
ENGAGE_SILENCE_TIMEOUT = 30.0  # 参与态持续沉默超此秒数 → 退出对话（回到需唤醒）
CLOSE_PHRASES = ("再见", "就这样", "先这样", "不聊了", "拜拜", "结束对话")


def model_dir() -> Path:
    import os
    return Path(os.environ.get(
        "ROBOT_REALTIME_MODEL_DIR",
        "/Users/yuanlw/Documents/huggingface/models/sherpa-onnx-realtime"))


def kws_dir() -> Path:
    return model_dir() / "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"


class RealtimeVoiceService:
    def __init__(self):
        self._lock = threading.Lock()
        self._spotter = None
        self._err = None

    def available(self) -> bool:
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        k = kws_dir()
        return (model_dir() / "silero_vad.onnx").is_file() and \
            (k / "tokens.txt").is_file() and \
            (k / "encoder-epoch-99-avg-1-chunk-16-left-64.onnx").is_file()

    def _keywords_file(self) -> Path:
        # 唤醒词拼音文件随模型目录存放（首次自动生成）
        p = model_dir() / "wake_keywords.txt"
        if not p.is_file():
            p.write_text(f"{WAKE_PINYIN} @{WAKE_LABEL}\n", encoding="utf-8")
        return p

    def spotter(self):
        import sherpa_onnx
        with self._lock:
            if self._spotter is None:
                k = kws_dir()
                self._spotter = sherpa_onnx.KeywordSpotter(
                    tokens=str(k / "tokens.txt"),
                    encoder=str(k / "encoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
                    decoder=str(k / "decoder-epoch-99-avg-1-chunk-16-left-64.onnx"),
                    joiner=str(k / "joiner-epoch-99-avg-1-chunk-16-left-64.onnx"),
                    keywords_file=str(self._keywords_file()),
                    keywords_threshold=0.25, keywords_score=1.5, num_threads=2)
            return self._spotter

    def create_session(self, wake_label: str = WAKE_LABEL) -> "VoiceSession":
        import sherpa_onnx
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model_dir() / "silero_vad.onnx")
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = VAD_MIN_SILENCE
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.window_size = 512
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        return VoiceSession(self.spotter(), vad, wake_label)


class VoiceSession:
    """单连接语音会话状态机（含长对话参与状态）。

    状态：idle（未参与，等唤醒词）/ engaged_listening（已参与，VAD 免唤醒听）/
    capturing（唤醒或打断后收录）/ processing / speaking。
    engaged 标志贯穿多轮：唤醒建立 → 每轮答完续期 → 持续沉默/明确收尾才退出。
    事件：wake / barge_in / endpoint / disengaged。
    """
    def __init__(self, spotter, vad, wake_label: str = WAKE_LABEL):
        self._spotter = spotter
        self._vad = vad
        self._wake_label = wake_label
        self._stream = spotter.create_stream()
        self.state = "idle"
        self.engaged = False
        self._last_activity = datetime.now().astimezone()
        self._prebuffer = deque(maxlen=int(PREBUFFER_SECONDS * SAMPLE_RATE))
        self._capture: list[np.ndarray] = []
        self._captured_samples = 0

    def _reset_idle(self):
        self._stream = self._spotter.create_stream()
        self._prebuffer.clear()
        self._capture = []
        self._captured_samples = 0
        self.state = "idle"

    def _reset_engaged_listening(self):
        self._vad.reset()
        self._capture = []
        self._captured_samples = 0
        self._last_activity = datetime.now().astimezone()
        self.state = "engaged_listening"

    def set_state(self, state: str):
        if state == "idle":
            self._reset_idle()
        elif state == "engaged_listening":
            self._reset_engaged_listening()
        elif state == "listening":  # 兼容旧调用：按参与标志归位
            self._reset_engaged_listening() if self.engaged else self._reset_idle()
        else:
            self.state = state

    def disengage(self):
        self.engaged = False
        self._reset_idle()

    def _endpoint(self, events):
        pcm = np.concatenate(self._capture) if self._capture else np.zeros(0, dtype=np.float32)
        self._capture = []
        self._captured_samples = 0
        if len(pcm) < int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE):
            self._vad.reset()
            return
        self._last_activity = datetime.now().astimezone()
        self.state = "processing"
        events.append({"type": "endpoint", "pcm": pcm})

    def _vad_capture(self, samples: np.ndarray, events: list) -> None:
        self._capture.append(samples)
        self._captured_samples += len(samples)
        self._vad.accept_waveform(samples)
        while not self._vad.empty():
            self._vad.front
            self._vad.pop()
            self._endpoint(events)
            break
        if self.state in ("capturing", "engaged_listening") and \
                self._captured_samples > int(MAX_CAPTURE_SECONDS * SAMPLE_RATE):
            self._endpoint(events)

    def _feed_kws(self, samples: np.ndarray, events: list, wake_type: str) -> None:
        self._prebuffer.extend(samples.tolist())
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream)
            if result and self._wake_label in result:
                self._capture = [np.array(self._prebuffer, dtype=np.float32)]
                self._captured_samples = len(self._prebuffer)
                self._vad.reset()
                self.engaged = True
                self._last_activity = datetime.now().astimezone()
                self.state = "capturing"
                events.append({"type": wake_type})
                return

    def accept(self, samples: np.ndarray) -> list[dict]:
        samples = np.asarray(samples, dtype=np.float32)
        events: list[dict] = []
        if self.state == "processing":
            return events
        if self.state == "idle":
            self._feed_kws(samples, events, "wake")
            return events
        if self.state == "speaking":
            self._feed_kws(samples, events, "barge_in")
            return events
        if self.state == "engaged_listening":
            now = datetime.now().astimezone()
            if (now - self._last_activity).total_seconds() > ENGAGE_SILENCE_TIMEOUT:
                self.engaged = False
                self._reset_idle()
                events.append({"type": "disengaged"})
                return events
            self._vad_capture(samples, events)
            return events
        if self.state == "capturing":
            self._vad_capture(samples, events)
            return events
        return events