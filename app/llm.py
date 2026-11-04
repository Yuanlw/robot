"""Bounded, streaming OpenAI-compatible model adapter."""
from __future__ import annotations

from typing import Any, Iterator

from .config import settings


class LLMService:
    def __init__(self) -> None:
        self._client: Any = None
        self._blocked_reason: str | None = None
        self.last_status = "not_configured"
        if settings.llm_api_key:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout_seconds,
                max_retries=0,
            )
            self.last_status = "configured"

    @property
    def enabled(self) -> bool:
        return self._client is not None and self._blocked_reason is None

    @property
    def blocked_reason(self) -> str | None:
        return self._blocked_reason

    def _request(self, text: str | list[dict[str, str]]) -> dict:
        messages = [
            {"role": "system", "content": (
                "你是家庭语音助手，用自然简短的中文回答，通常一到三句话。"
                "不要输出Markdown、代码或思考过程。当前应用只接通了聊天、规则提醒和已录入人脸识别；"
                "没有天气、搜索、音乐、家电控制工具，不能声称能执行这些操作。提醒必须经过应用确认并保存；"
                "你没有创建提醒的工具，不能说已经创建、已经记下或保证会提醒。"
                "需要设置提醒时请引导使用‘提醒我明天晚上八点练琴’的格式。"
            )}
        ]
        if isinstance(text, str):
            messages.append({"role": "user", "content": text})
        else:
            messages.extend(text)
        args = {
            "model": settings.llm_model,
            "max_tokens": settings.llm_max_tokens,
            "messages": messages,
            "stream": True,
        }
        extra = {}
        if settings.llm_thinking is not None:
            # Bailian compatible-mode: thinking is on by default for streaming;
            # qwen takes a boolean enable_thinking, unlike the old "thinking" object.
            extra["enable_thinking"] = settings.llm_thinking == "enabled"
        if settings.llm_reasoning_effort and settings.llm_thinking != "disabled":
            extra["reasoning_effort"] = settings.llm_reasoning_effort
        if extra:
            args["extra_body"] = extra
        return args

    def stream_chat(self, text: str) -> Iterator[str]:
        yield from self.stream_messages([{"role": "user", "content": text}])

    def stream_messages(self, messages: list[dict[str, str]]) -> Iterator[str]:
        if not self.enabled:
            raise RuntimeError(self._blocked_reason or "聊天模型尚未配置，请先配置可用的模型服务。")
        received = False
        try:
            with self._client.chat.completions.create(**self._request(messages)) as stream:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        received = True
                        yield chunk.choices[0].delta.content
            if not received:
                raise RuntimeError("模型没有返回回答，请稍后重试。")
            self.last_status = "available"
        except Exception as exc:
            from openai import APITimeoutError
            self.last_status = "error"
            if isinstance(exc, APITimeoutError):
                self.last_status = "timeout"
                raise RuntimeError("聊天模型响应超时，请重试或更换更快的模型服务。") from exc
            if getattr(exc, "status_code", None) in (401, 403):
                self._blocked_reason = "模型服务拒绝访问，请检查密钥、模型权限及是否允许第三方 API 调用。"
                raise RuntimeError(self._blocked_reason) from exc
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("聊天模型暂时不可用，请检查模型服务或网络。") from exc

    def summarize(self, previous_summary: str, messages: list[dict[str, str]]) -> str:
        """Compress older turns into a concise factual memory for this conversation."""
        if not self.enabled:
            raise RuntimeError(self._blocked_reason or "聊天模型暂时不可用")
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        try:
            response = self._client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=220,
                stream=False,
                messages=[
                    {"role": "system", "content": "把家庭助手的旧对话压缩为简短记忆，保留用户明确提供的身份称呼、稳定偏好、持续目标、未完成事项和必要事实。不要推测，不保留寒暄和一次性闲聊，使用第三人称描述用户。最多 180 个汉字，只返回摘要。"},
                    {"role": "user", "content": f"已有记忆：{previous_summary or '无'}\n\n较早对话：\n{transcript}"},
                ],
            )
            summary = (response.choices[0].message.content or "").strip()
            if not summary:
                raise RuntimeError("记忆摘要为空")
            return summary[:600]
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError("对话记忆整理失败") from exc

    def chat(self, text: str) -> str:
        return "".join(self.stream_chat(text)).strip()
