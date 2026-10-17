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
        self._blocked_reason: str | None = None
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
        return self._client is not None and self._blocked_reason is None

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    def chat(self, text: str) -> str:
        if not self._client:
            raise RuntimeError("LLM 未配置：请设置 ROBOT_LLM_API_KEY 并安装 openai")
        if self._blocked_reason:
            raise RuntimeError(self._blocked_reason)
        try:
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
        except Exception as exc:
            if getattr(exc, "status_code", None) == 403 or "official clients" in str(exc):
                self._blocked_reason = "模型服务拒绝了家庭机器人请求：此 Key 目前只允许官方 Codex 客户端。请让中间商把 Key 切换为允许第三方 OpenAI 兼容 API 的类型"
                raise RuntimeError(self._blocked_reason) from exc
            raise
        return (response.choices[0].message.content or "").strip()
