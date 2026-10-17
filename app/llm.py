from __future__ import annotations

from typing import Any

from .config import settings


class LLMService:
    """Optional OpenAI-compatible model adapter.

    The robot remains usable without a model: local command parsing handles
    the common reminder form and this adapter is only used as a fallback.
    """

    def __init__(self) -> None:
        self._client: Any = None
        if settings.llm_api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    timeout=settings.llm_timeout_seconds,
                )
            except ImportError:
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def chat(self, text: str) -> str:
        if not self._client:
            raise RuntimeError("LLM 未配置：请设置 ROBOT_LLM_API_KEY 并安装 openai")
        response = self._client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": "你是家庭机器人助手。用简洁自然的中文回答，涉及提醒时先确认时间和内容。",
                },
                {"role": "user", "content": text},
            ],
        )
        return (response.choices[0].message.content or "").strip()
