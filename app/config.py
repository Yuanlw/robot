from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    timezone: str = "Asia/Shanghai"
    face_threshold: float = 0.48
    lbph_threshold: float = 75.0
    # 人脸增量学习与样本池（2026-11-13）
    face_learn: bool = True
    face_learn_threshold: float = 0.40
    face_max_samples: int = 8
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
    
    # 学习陪伴模式（2026-11-11 新增，默认关闭）
    study_enabled: bool = False
    study_focus_minutes: int = 40
    study_break_minutes: int = 5
    study_max_reminders_per_hour: int = 6

    # 坐姿守护（第二期 2A：默认关闭；画面仅本机内存分析，不落盘不上传）
    vision_enabled: bool = False
    vision_sample_seconds: int = 5
    vision_max_bytes: int = 2 * 1024 * 1024

    model_config = SettingsConfigDict(env_prefix="ROBOT_", env_file=".env")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
