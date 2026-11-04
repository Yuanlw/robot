from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    timezone: str = "Asia/Shanghai"
    face_threshold: float = 0.48
    lbph_threshold: float = 75.0
    llm_api_key: str | None = None
    llm_base_url: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    llm_model: str = "qwen3.8-flash"
    llm_timeout_seconds: float = 12.0
    llm_max_tokens: int = 160
    llm_reasoning_effort: str = ""
    llm_thinking: Literal["enabled", "disabled"] | None = None
    agent_backend: Literal["harness", "legacy"] = "harness"
    harness_command: str = "/usr/local/bin/dsh"
    harness_timeout_seconds: float = 60.0
    harness_max_tokens: int = 700
    harness_idle_seconds: float = 300.0
    harness_max_warm: int = 4
    stt_backend: Literal["dashscope", "browser"] = "dashscope"
    stt_model: str = "qwen3-asr-flash"
    tts_backend: str = "auto"
    tts_voice: str = "bailian:Cherry"
    tts_speed: float = 1.0
    tts_system_voice: str = "Tingting"
    tts_model_dir: Path | None = None
    tts_bailian_model: str = "qwen3-tts-flash"
    tts_bailian_voice: str = "Cherry"

    model_config = SettingsConfigDict(env_prefix="ROBOT_", env_file=".env")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
