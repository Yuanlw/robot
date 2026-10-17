from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_dir: Path = Path("data")
    timezone: str = "Asia/Shanghai"
    face_threshold: float = 0.48

    model_config = SettingsConfigDict(env_prefix="ROBOT_", env_file=".env")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
