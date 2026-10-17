"""Voice adapters with browser input and native OS speech output fallbacks."""

import shutil
import subprocess


class VoiceService:
    def status(self) -> str:
        if shutil.which("say"):
            return "macos-say"
        try:
            import pyttsx3  # noqa: F401
            return "local-tts"
        except ImportError:
            if shutil.which("say"):
                return "macos-say"
            if shutil.which("espeak-ng") or shutil.which("espeak"):
                return "espeak"
            return "text-only"

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
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            return "spoken"
        except Exception:
            # pyttsx3 can import successfully but fail to initialize its macOS
            # driver. Fall back to the system voice command in that case.
            command = shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak")
            if not command:
                return "text-only"
            args = [command, text] if command.endswith("say") else [command, "-v", "cmn", text]
            try:
                subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return "spoken"
            except OSError:
                return "text-only"
