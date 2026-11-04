"""Local Mandarin neural speech with an explicit Chinese system fallback."""
import io
import re
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
import json
import os
import json
import os

from .config import settings


KOKORO_VOICE_CHOICES = [
    {"id": "zf_001", "label": "Kokoro · 当前默认女声"},
]
KOKORO_VOICE_IDS = {item["id"] for item in KOKORO_VOICE_CHOICES}
# 百炼云端音色（qwen3-tts-flash）。demo 2026-11-03 实测：厂商首片 314~379ms，
# 整句 RTF 0.25~0.36，快于实时 3~4 倍。只列实测过的 Cherry，其余音色验证后再加。
BAILIAN_VOICE_CHOICES = [
    {"id": "bailian:Cherry", "label": "百炼 · Cherry 云端女声"},
]
BAILIAN_VOICE_IDS = {item["id"] for item in BAILIAN_VOICE_CHOICES}
ARGMAX_VOICE_CHOICES = [
    {"id": "argmax:serena", "label": "Argmax · Serena 温柔女声"},
    {"id": "argmax:vivian", "label": "Argmax · Vivian 明亮女声"},
    {"id": "argmax:dylan", "label": "Argmax · Dylan 北京男声"},
    {"id": "argmax:eric", "label": "Argmax · Eric 成都男声"},
    {"id": "argmax:uncle-fu", "label": "Argmax · Uncle Fu 低沉男声"},
]
ARGMAX_VOICE_IDS = {item["id"] for item in ARGMAX_VOICE_CHOICES}


def spoken_text(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "，这里包含代码，请查看屏幕。", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s*[#>*]+\s*", "", text)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"(?<!\d)([01]?\d|2[0-3]):([0-5]\d)(?!\d)",
                  lambda m: f"{int(m[1])}点" + (f"{int(m[2])}分" if int(m[2]) else "整"), text)
    return text.strip()


class VoiceService:
    def __init__(self):
        self._model = None
        self._g2p = None
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self._neural_error = None
        self._argmax_error = None
        self._bailian_error = None
        self.last_bailian_first_chunk_ms = None
        self._load_voice_preference()

    @property
    def preference_path(self) -> Path:
        return settings.data_dir / "voice-preference.json"

    def _load_voice_preference(self) -> None:
        try:
            value = json.loads(self.preference_path.read_text()).get("voice")
            if value in KOKORO_VOICE_IDS | ARGMAX_VOICE_IDS | BAILIAN_VOICE_IDS:
                settings.tts_voice = value
        except (OSError, ValueError, AttributeError):
            pass

    def set_voice(self, voice: str) -> None:
        if voice not in KOKORO_VOICE_IDS | ARGMAX_VOICE_IDS | BAILIAN_VOICE_IDS:
            raise ValueError("请选择列表中的中文音色。")
        if voice in KOKORO_VOICE_IDS and not self.neural_available():
            raise RuntimeError("本地 Kokoro 中文语音尚未就绪。")
        if voice in ARGMAX_VOICE_IDS and not self.argmax_available():
            raise RuntimeError("Argmax 0.6B 尚未就绪，请检查命令行程序和模型。")
        if voice in BAILIAN_VOICE_IDS and not self.bailian_available():
            raise RuntimeError("未配置百炼 API Key，无法使用云端音色。")
        with self._lock:
            self.preference_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.preference_path.with_suffix(".tmp")
            temp_path.write_text(json.dumps({"voice": voice}, ensure_ascii=False))
            os.replace(temp_path, self.preference_path)
            settings.tts_voice = voice
            self._argmax_error = None
            self._bailian_error = None
            self._cache.clear()

    @property
    def model_dir(self) -> Path:
        return settings.tts_model_dir or settings.data_dir / "tts"

    def neural_available(self) -> bool:
        import importlib.util
        return all((self.model_dir / name).is_file() for name in (
            "kokoro-v1.1-zh.onnx", "voices-v1.1-zh.bin", "config.json"
        )) and all(importlib.util.find_spec(name) is not None for name in ("kokoro_onnx", "misaki", "soundfile"))

    def bailian_available(self) -> bool:
        return bool(settings.llm_api_key)

    @property
    def argmax_cli(self) -> Path:
        return Path(os.environ.get("ROBOT_ARGMAX_CLI", "/Users/yuanlw/Documents/ChatGPT/智能机器人/argmax-oss-swift/.build/release/argmax-cli"))

    @property
    def argmax_model_dir(self) -> Path:
        return Path(os.environ.get("ROBOT_ARGMAX_MODEL_DIR", "/Users/yuanlw/Documents/huggingface/models/argmaxinc/ttskit-coreml"))

    def argmax_available(self) -> bool:
        return self.argmax_cli.is_file() and os.access(self.argmax_cli, os.X_OK) and self.argmax_model_dir.is_dir()

    def status(self) -> str:
        if settings.tts_voice in BAILIAN_VOICE_IDS and self.bailian_available() and not self._bailian_error:
            return "bailian-zh-cloud"
        if settings.tts_voice in ARGMAX_VOICE_IDS and self.argmax_available() and not self._argmax_error:
            return "argmax-zh-local"
        if settings.tts_backend != "system" and self.neural_available() and not self._neural_error:
            return "kokoro-zh-local"
        if shutil.which("say") and shutil.which("afconvert"):
            return "macos-zh-system"
        return "browser-zh"

    def details(self) -> dict:
        backend = self.status()
        choices = ((BAILIAN_VOICE_CHOICES if self.bailian_available() else [])
                   + (KOKORO_VOICE_CHOICES if self.neural_available() else [])
                   + (ARGMAX_VOICE_CHOICES if self.argmax_available() else []))
        # 返回的当前音色必须在可选列表内，否则设置页下拉框会选中空值；
        # 所选后端不可用时回退到第一个可用音色（与 synthesize 的降级链一致）。
        available_ids = {item["id"] for item in choices}
        voice = settings.tts_voice if settings.tts_voice in available_ids else (
            choices[0]["id"] if choices else settings.tts_system_voice)
        return {"backend": backend, "voice": voice,
                "available_voices": choices,
                "ready": backend != "browser-zh",
                "fallback_reason": self._bailian_error or self._argmax_error or self._neural_error}

    def warmup(self):
        # 预热当前音色链路（含云端）；失败不阻塞启动，合成时链上自动回退。
        try:
            if self.status() != "browser-zh":
                self.synthesize("你好。")
        except Exception:
            pass

    def _neural(self, text: str) -> bytes:
        import soundfile as sf
        from kokoro_onnx import Kokoro
        from misaki import zh
        if self._model is None:
            self._g2p = zh.ZHG2P(version="1.1")
            self._model = Kokoro(str(self.model_dir / "kokoro-v1.1-zh.onnx"),
                                 str(self.model_dir / "voices-v1.1-zh.bin"),
                                 vocab_config=str(self.model_dir / "config.json"))
        phonemes, _ = self._g2p(text)
        voice = settings.tts_voice if settings.tts_voice in KOKORO_VOICE_IDS else "zf_001"
        samples, rate = self._model.create(phonemes, voice=voice,
                                           speed=settings.tts_speed, is_phonemes=True)
        if len(samples) == 0:
            raise RuntimeError("语音合成没有生成声音。")
        output = io.BytesIO()
        sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
        return output.getvalue()

    def _argmax(self, text: str, speaker: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="robot-argmax-") as tmp:
            output = Path(tmp) / "speech.wav"
            subprocess.run([
                str(self.argmax_cli), "tts", "--text", text, "--speaker", speaker,
                "--language", "chinese", "--model", "0.6b", "--models-path", str(self.argmax_model_dir),
                "--output-path", str(Path(tmp) / "speech"), "--output-format", "wav",
            ], check=True, timeout=90, capture_output=True)
            if not output.is_file():
                raise RuntimeError("Argmax 未生成音频。")
            wav = Path(tmp) / "speech-pcm.wav"
            subprocess.run(["/usr/local/bin/ffmpeg", "-v", "error", "-i", str(output),
                            "-c:a", "pcm_s16le", str(wav)], check=True, timeout=15, capture_output=True)
            return wav.read_bytes()

    def _bailian(self, text: str, voice_name: str) -> bytes:
        from . import bailian
        wav, first_ms = bailian.tts_synthesize(text, voice=voice_name)
        self.last_bailian_first_chunk_ms = first_ms
        return wav

    def _neural_fallback(self, text: str) -> bytes:
        selected = settings.tts_voice
        try:
            settings.tts_voice = "zf_001"
            return self._neural(text)
        finally:
            settings.tts_voice = selected

    def _system(self, text: str) -> bytes:
        if not shutil.which("say") or not shutil.which("afconvert"):
            raise RuntimeError("本地语音不可用，请安装中文语音模型或启用浏览器中文语音。")
        with tempfile.TemporaryDirectory(prefix="robot-tts-") as tmp:
            aiff, wav = Path(tmp) / "speech.aiff", Path(tmp) / "speech.wav"
            subprocess.run(["say", "-v", settings.tts_system_voice, "-o", str(aiff), text],
                           check=True, timeout=20, capture_output=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16", str(aiff), str(wav)],
                           check=True, timeout=10, capture_output=True)
            return wav.read_bytes()

    def synthesize(self, text: str) -> bytes:
        text = spoken_text(text)
        if not text:
            raise ValueError("请输入需要朗读的文字。")
        with self._lock:
            key = (text, settings.tts_backend, settings.tts_voice, settings.tts_speed, settings.tts_system_voice)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if settings.tts_voice in BAILIAN_VOICE_IDS and self.bailian_available() and not self._bailian_error:
                try:
                    audio = self._bailian(text, settings.tts_voice.split(":", 1)[1])
                except Exception:
                    self._bailian_error = "百炼云端语音合成失败，已回退到本地语音。"
                    if self.neural_available():
                        audio = self._neural_fallback(text)
                    else:
                        audio = self._system(text)
            elif settings.tts_voice in ARGMAX_VOICE_IDS and self.argmax_available() and not self._argmax_error:
                try:
                    audio = self._argmax(text, settings.tts_voice.split(":", 1)[1])
                except (OSError, subprocess.SubprocessError, RuntimeError):
                    self._argmax_error = "Argmax 中文语音合成失败，已回退到 Kokoro。"
                    if self.neural_available():
                        audio = self._neural_fallback(text)
                    else:
                        audio = self._system(text)
            elif settings.tts_voice in ARGMAX_VOICE_IDS and self.neural_available():
                audio = self._neural_fallback(text)
            elif settings.tts_backend != "system" and self.neural_available() and not self._neural_error:
                try:
                    audio = self._neural(text)
                except Exception:
                    self._neural_error = "中文神经语音初始化或合成失败。"
                    if settings.tts_backend == "kokoro":
                        raise RuntimeError(self._neural_error)
                    audio = self._system(text)
            else:
                if settings.tts_backend == "kokoro":
                    raise RuntimeError("中文神经语音尚未就绪，请检查模型文件和语音依赖。")
                audio = self._system(text)
            self._cache[key] = audio
            while len(self._cache) > 16:
                self._cache.popitem(last=False)
            return audio

    def transcribe(self, audio_path: str) -> str:
        try:
            import speech_recognition as sr
        except ImportError as exc:
            raise RuntimeError("语音识别未安装，请安装 SpeechRecognition 和对应音频依赖") from exc
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio, language="zh-CN")

    def speak(self, text: str) -> str:
        audio = self.synthesize(text)
        if not shutil.which("afplay"):
            return "text-only"
        with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
            wav.write(audio)
            wav.flush()
            subprocess.run(["afplay", wav.name], check=True, timeout=120, capture_output=True)
        return "spoken"
