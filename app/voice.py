"""Voice adapters. MVP stays usable when optional packages are absent."""


class VoiceService:
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
        except ImportError:
            return "text-only"
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return "spoken"
