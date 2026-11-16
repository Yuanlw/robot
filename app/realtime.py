"""学习模式纯语音对话引擎（阶段 B 第一步）：常开麦 + 唤醒词 + VAD 自动断句。

设计（半双工，免按钮）：
- listening：麦克风常开，音频喂 KWS 等唤醒词"阿伴阿伴"；
- capturing：听到唤醒词后收录这句（含唤醒前 ~1s 预缓冲），VAD 检测说完（静音）→ endpoint；
- processing/speaking：由 WebSocket 处理层置位；期间忽略麦克风（半双工防回声自唤醒）。

隐私红线：音频只在内存流过本地 sherpa-onnx 模型，不录、不存、不上传；
唤醒后的语音按现有 /api/stt 隐私边界送百炼识别。

纯状态机 + 本地模型，可用合成/自带测试音频单测，不依赖真实麦克风。
sherpa-onnx 或模型缺失时 available()=False，功能整体优雅降级（回退按钮模式）。
"""
import threading
from collections import deque
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
    """单连接语音会话状态机。accept(samples) 返回事件列表，处理层据此驱动 STT/Agent/TTS。

    事件：{'type':'wake'} / {'type':'endpoint','pcm':np.ndarray} / {'type':'state','state':...}
    处理层在 endpoint 后置 state='processing'，播放时置 'speaking'，播完置 'listening'。
    """
    def __init__(self, spotter, vad, wake_label: str = WAKE_LABEL):
        self._spotter = spotter
        self._vad = vad
        self._wake_label = wake_label
        self._stream = spotter.create_stream()
        self.state = "listening"
        self._prebuffer = deque(maxlen=int(PREBUFFER_SECONDS * SAMPLE_RATE))
        self._capture: list[np.ndarray] = []
        self._captured_samples = 0

    def _reset_listening(self):
        self._stream = self._spotter.create_stream()
        self._prebuffer.clear()
        self._capture = []
        self._captured_samples = 0
        self.state = "listening"

    def set_state(self, state: str):
        """处理层切换 processing/speaking/listening。回到 listening 时重置 KWS 流。"""
        if state == "listening":
            self._reset_listening()
        else:
            self.state = state

    def _endpoint(self, events):
        pcm = np.concatenate(self._capture) if self._capture else np.zeros(0, dtype=np.float32)
        self._capture = []
        self._captured_samples = 0
        # 太短（多半只是唤醒词或噪声）：丢弃，继续收录
        if len(pcm) < int(MIN_UTTERANCE_SECONDS * SAMPLE_RATE):
            self._vad.reset()
            return
        self.state = "processing"
        events.append({"type": "endpoint", "pcm": pcm})

    def accept(self, samples: np.ndarray) -> list[dict]:
        samples = np.asarray(samples, dtype=np.float32)
        events: list[dict] = []
        # 半双工：处理/播放期间忽略麦克风，防机器人自己的声音被采回自唤醒
        if self.state in ("processing", "speaking"):
            return events

        if self.state == "listening":
            self._prebuffer.extend(samples.tolist())
            self._stream.accept_waveform(SAMPLE_RATE, samples)
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
                result = self._spotter.get_result(self._stream)
                if result and self._wake_label in result:
                    # 唤醒：带预缓冲进入收录，重置 VAD
                    self._capture = [np.array(self._prebuffer, dtype=np.float32)]
                    self._captured_samples = len(self._prebuffer)
                    self._vad.reset()
                    self.state = "capturing"
                    events.append({"type": "wake"})
                    break
            return events

        if self.state == "capturing":
            self._capture.append(samples)
            self._captured_samples += len(samples)
            self._vad.accept_waveform(samples)
            while not self._vad.empty():
                self._vad.front  # 取出段（触发断句）
                self._vad.pop()
                self._endpoint(events)
                break
            if self.state == "capturing" and self._captured_samples > int(MAX_CAPTURE_SECONDS * SAMPLE_RATE):
                self._endpoint(events)
            return events

        return events