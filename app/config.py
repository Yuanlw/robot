from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    timezone: str = "Asia/Shanghai"
    face_threshold: float = 0.48
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.aigocode.dev/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 20.0

    model_config = SettingsConfigDict(env_prefix="ROBOT_", env_file=".env")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
