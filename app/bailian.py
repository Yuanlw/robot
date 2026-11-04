"""百炼（DashScope）STT/TTS 客户端。

从 voice-poc demo（2026-11-03 实测验证）移植，协议要点保真：
- ASR  qwen3-asr-flash 走 OpenAI 兼容 chat/completions（不是 multimodal-generation，
  打错端点会返回 HTTP 400 空 body，极难排查），音频用 base64 data URI 直传，无需 OSS。
  响应空文本是合法结果（静音/纯噪声返回 ""），响应结构不对才是真故障。
- TTS  qwen3-tts-flash 走 DashScope 原生 multimodal-generation + SSE。
  中间 chunk 是 raw PCM 24kHz/16bit/mono，浏览器放不了裸 PCM，收齐后服务端包成 WAV。
  单次输入 ≤600 字符；language_type 指定 Chinese 的合成质量显著优于 Auto（文档原文）。
  必须快速失败：demo 实测出现过单次合成挂到 120s 读超时，read 超时取 20s，重试 1 次。
- 静默门控 is_silence：LLM 架构的 ASR 喂全零静音不会返回空，而是幻觉出文本
  （demo 实测：1.3s 全零静音被识别成整段词表）。宁可判"没听到"，不能把静音交给模型。

零音频编解码依赖：WAV 封装用 stdlib wave（本机 libsndfile 是 x86_64、与 arm64
python 不兼容，soundfile/librosa 会直接 OSError——demo README 明确警告过）。
"""
from __future__ import annotations

import base64
import io
import json
import struct
import time
import wave

import httpx

from .config import settings

MM_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
CC_PATH = "/compatible-mode/v1/chat/completions"
TIMEOUT_STT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)
# TTS 必须快速失败（demo 实测有单次挂到 120s 读超时的案例）：
# read 是「两个 chunk 之间」的间隔上限，正常首片 300~400ms，20s 已极宽松。
TIMEOUT_TTS = httpx.Timeout(connect=8.0, read=20.0, write=20.0, pool=10.0)
TTS_RETRY = 1
TTS_PCM_RATE = 24000
TTS_MAX_CHARS = 600

# 静音判定阈值（demo 2026-11-03 实测：麦克风哑掉时落盘音频逐采样点全零，
# RMS=0/峰值=0/有声帧 0%；正常人声 RMS≈2080/峰值≈22483/有声帧 34%。
# 真实麦克风再安静也有底噪 RMS>100，「峰值<64」基本只可能是数字静音）。
_SILENT_PEAK = 64
_SILENT_VOICED = 0.01


class BailianError(RuntimeError):
    pass


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """16bit 单声道小端 PCM -> WAV（stdlib，不依赖 soundfile）。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def audio_stats(pcm: bytes, sample_rate: int = 16000) -> dict:
    """16bit 单声道 PCM 的电平统计（纯 stdlib，逐字移植 demo pipeline.audio_stats）。"""
    n = len(pcm) // 2
    if n == 0:
        return {"samples": 0, "seconds": 0.0, "rms": 0.0, "peak": 0, "voiced_ratio": 0.0}
    smp = struct.unpack(f"<{n}h", pcm[: n * 2])
    peak = 0
    acc = 0
    for v in smp:
        a = -v if v < 0 else v
        if a > peak:
            peak = a
        acc += v * v
    rms = (acc / n) ** 0.5
    win = max(int(sample_rate * 0.02), 1)          # 20ms 一帧
    voiced = 0
    total = 0
    for i in range(0, n - win + 1, win):
        seg = smp[i:i + win]
        fr = (sum(v * v for v in seg) / len(seg)) ** 0.5
        total += 1
        if fr > 300:
            voiced += 1
    return {
        "samples": n, "seconds": round(n / max(sample_rate, 1), 2),
        "rms": round(rms, 1), "peak": peak,
        "voiced_ratio": round(voiced / total, 3) if total else 0.0,
    }


def is_silence(pcm: bytes, sample_rate: int = 16000) -> tuple[bool, dict]:
    """返回 (是否静音, 电平统计)。静音必须在送 ASR 之前拦下，见模块 docstring。"""
    st = audio_stats(pcm, sample_rate)
    silent = st["samples"] == 0 or (
        st["peak"] < _SILENT_PEAK and st["voiced_ratio"] < _SILENT_VOICED
    )
    return silent, st


def _host(base_url: str) -> str:
    """从 compatible-mode base_url 还原 DashScope 原生协议的域名根。"""
    b = (base_url or "").rstrip("/")
    for suffix in ("/compatible-mode/v1", "/v1"):
        if b.endswith(suffix):
            return b[: -len(suffix)]
    return b


def _cc_url(base_url: str) -> str:
    """ASR 走 OpenAI 兼容 chat/completions；base_url 可只填域名或填到 /v1。"""
    b = (base_url or "").rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + CC_PATH


def _headers(key: str, sse: bool) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable" if sse else "disable",
    }


def _err(status: int, body: str, what: str) -> BailianError:
    b = body[:600].replace("\n", " ")
    hint = ""
    if status in (401, 403):
        hint = "（鉴权失败：检查百炼 API Key，注意北京/新加坡地域 key 不通用）"
    elif status == 404:
        hint = "（路由不存在：model 名是否写错，或该模型不走这个端点）"
    elif status == 429:
        hint = "（限流：qwen3-asr-flash 文档标注 RPM 100）"
    return BailianError(f"{what} HTTP {status}{hint}: {b}")


def stt_transcribe(pcm: bytes, sample_rate: int = 16000, *,
                   api_key: str | None = None, base_url: str | None = None,
                   model: str | None = None, transport: httpx.BaseTransport | None = None) -> str:
    """整句识别（说完再识别），与产品的按钮式录音交互一致。

    language 不指定（demo 实测：不指定更稳也更快，409ms vs 466ms；
    指定 zh 只覆盖普通话/四川话/闽南语/吴语，粤语要单独 yue）。
    领域词表默认关闭（demo 2026-11-03 A/B 实测零收益，且静音时会把整段
    词表幻觉回来，纯负担）。
    """
    key = api_key or settings.llm_api_key
    if not key:
        raise BailianError("未配置百炼 API Key，无法使用云端语音识别。")
    if not pcm:
        raise BailianError("语音识别收到空音频。")
    wav = pcm_to_wav(pcm, sample_rate)
    uri = "data:audio/wav;base64," + base64.b64encode(wav).decode()
    payload = {
        "model": model or settings.stt_model,
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": uri}}]}],
        "stream": False,
    }
    client_kwargs = {"timeout": TIMEOUT_STT}
    if transport is not None:
        client_kwargs["transport"] = transport
    with httpx.Client(**client_kwargs) as c:
        r = c.post(_cc_url(base_url or settings.llm_base_url),
                   headers=_headers(key, sse=False), json=payload)
    if r.status_code != 200:
        raise _err(r.status_code, r.text, "百炼语音识别")
    try:
        j = r.json()
    except Exception:
        raise BailianError(f"百炼语音识别返回非 JSON：{r.text[:300]}")

    # 标准 OpenAI 形态：choices[0].message.content
    txt = ""
    choices = j.get("choices")
    if isinstance(choices, list) and choices:
        cont = ((choices[0] or {}).get("message") or {}).get("content")
        if isinstance(cont, str):
            txt = cont
        elif isinstance(cont, list):
            txt = "".join(x.get("text", "") for x in cont if isinstance(x, dict))
    if not txt:
        # 兜底：DashScope 原生形态 output.sentence.text / output.text
        out = j.get("output") or {}
        sent = out.get("sentence")
        if isinstance(sent, dict):
            txt = sent.get("text") or ""
        txt = txt or out.get("text") or ""
    if not txt:
        # ★空文本是合法结果，不是错误：静音、纯噪声、非人声都会返回 ""。
        #   但「响应结构不对」（连 choices 都没有）才是真故障，两者必须区分。
        if isinstance(choices, list) and choices:
            return ""
        raise BailianError(
            f"百炼语音识别响应结构异常 code={j.get('code') or ''} "
            f"message={j.get('message') or ''} raw={json.dumps(j, ensure_ascii=False)[:300]}"
        )
    return str(txt).strip()


def tts_synthesize(text: str, *, voice: str | None = None,
                   api_key: str | None = None, base_url: str | None = None,
                   model: str | None = None, transport: httpx.BaseTransport | None = None) -> tuple[bytes, float | None]:
    """整句合成，返回 (WAV bytes, 厂商首片耗时 ms)。

    实现折中与 demo 一致：SSE 中间 chunk 是 raw PCM，浏览器 <audio> 放不了，
    收齐整句 PCM 后包成一个 WAV。产品端已按句切分（每句二三十字），可接受。
    """
    key = api_key or settings.llm_api_key
    if not key:
        raise BailianError("未配置百炼 API Key，无法使用云端语音合成。")
    text = (text or "").strip()
    if not text:
        raise BailianError("请输入需要朗读的文字。")
    if len(text) > TTS_MAX_CHARS:
        text = text[:TTS_MAX_CHARS]  # 文档：单次最大输入 600 字符
    payload = {
        "model": model or settings.tts_bailian_model,
        "input": {"text": text, "voice": voice or settings.tts_bailian_voice,
                  "language_type": "Chinese"},
    }
    url = _host(base_url or settings.llm_base_url) + MM_PATH
    client_kwargs = {"timeout": TIMEOUT_TTS}
    if transport is not None:
        client_kwargs["transport"] = transport

    pcm = bytearray()
    first_chunk_ms: float | None = None
    last_err: Exception | None = None
    for attempt in range(TTS_RETRY + 1):
        first_chunk_ms = None
        pcm = bytearray()
        t0 = time.perf_counter()
        try:
            with httpx.Client(**client_kwargs) as c:
                with c.stream("POST", url, headers=_headers(key, sse=True), json=payload) as r:
                    if r.status_code != 200:
                        r.read()
                        raise _err(r.status_code, r.text, "百炼语音合成")
                    for line in r.iter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(obj, dict):
                            continue
                        if obj.get("code"):
                            # 参数类错误重试没意义，直接抛
                            raise BailianError(f"百炼语音合成 code={obj.get('code')} message={obj.get('message')}")
                        audio = (obj.get("output") or {}).get("audio") or {}
                        b64 = audio.get("data")
                        if b64:
                            if first_chunk_ms is None:
                                first_chunk_ms = round((time.perf_counter() - t0) * 1000, 1)
                            try:
                                pcm += base64.b64decode(b64)
                            except Exception as e:
                                raise BailianError(f"百炼语音合成 Base64 解码失败：{e}")
            if not pcm:
                raise BailianError("百炼语音合成流结束但没收到任何音频片段")
            break  # 成功
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
            if attempt >= TTS_RETRY:
                raise BailianError(
                    f"百炼语音合成超时/网络失败（已重试 {attempt} 次，读超时 {TIMEOUT_TTS.read}s）：{type(e).__name__}"
                ) from e
            continue
        except BailianError as e:
            # 只重试 429/5xx；4xx 是参数问题，重试纯属浪费
            if any(f"HTTP {c}" in str(e) for c in (429, 500, 502, 503, 504)) and attempt < TTS_RETRY:
                last_err = e
                continue
            raise
    if last_err and not pcm:
        raise BailianError(f"百炼语音合成最终失败：{last_err}")
    return pcm_to_wav(bytes(pcm), TTS_PCM_RATE), first_chunk_ms
