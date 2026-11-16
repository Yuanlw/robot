"""真实链路语音冒烟：服务重启后、任何对话之前运行，第一句必须出声。

背景（2026-11-14 静默无声事故）：157 个单测全绿 + 接口 200，但浏览器第一句
完全无声——测试里语音全是 mock，e2e 走的是服务端 afplay 另一条路，且故障
锁存后复测全是热态。本脚本打真实运行中的服务，补上这条盲区。

用法：
    ./run.sh 启动服务后立刻执行（不要先在页面说话）：
    .venv-voice/bin/python scripts/smoke_voice.py
    可选：--base-url http://127.0.0.1:8000  --timeout 20

断言三条，任一不满足即 FAIL（退出码 1）：
  1. 第一句 /api/voice/audio 在 --timeout 秒内返回 200 的 WAV；
  2. 音频非静音（RMS > 100，时长 > 0.3 秒）；
  3. 合成后 fallback_reason 为空（所选音色真的在用，没有偷偷降级）。
"""
import argparse
import array
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
import wave


def request(base, path, payload=None, timeout=20):
    req = urllib.request.Request(base + path)
    req.add_header("X-Robot-Client", "smoke")  # CSRF 中间件要求
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        return resp.status, resp.read()


def wav_stats(data):
    with wave.open(io.BytesIO(data)) as w:
        seconds = w.getnframes() / w.getframerate()
        samples = array.array("h")
        samples.frombytes(w.readframes(w.getnframes()))
    rms = math.sqrt(sum(s * s for s in samples) / max(len(samples), 1))
    return seconds, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="第一句合成上限，须与前端预取窗口一致（默认 20 秒）")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    failures = []

    try:
        status, _ = request(base, "/api/health", timeout=5)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print("FAIL: 服务不在跑（%s），先 ./run.sh" % exc)
        return 1
    if status != 200:
        print("FAIL: 服务健康检查返回 %s，先 ./run.sh" % status)
        return 1
    _, raw = request(base, "/api/voice/settings", timeout=5)
    before = json.loads(raw)
    print("音色=%s 后端=%s" % (before["voice"], before["backend"]))

    text = "你好，我在这里。"
    started = time.monotonic()
    try:
        status, audio = request(base, "/api/voice/audio", {"text": text}, timeout=args.timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print("FAIL: 第一句 %.1f 秒内未返回（%s）——正是 2026-11-14 事故的形态"
              % (time.monotonic() - started, exc))
        return 1
    elapsed = time.monotonic() - started
    if status != 200 or not audio.startswith(b"RIFF"):
        failures.append("第一句返回 %s，不是 WAV" % status)
    else:
        seconds, rms = wav_stats(audio)
        print("第一句 %.1f 秒返回，音频 %.1f 秒，RMS=%.0f" % (elapsed, seconds, rms))
        if seconds <= 0.3 or rms <= 100:
            failures.append("音频疑似静音（时长 %.2fs RMS %.0f）" % (seconds, rms))

    _, raw = request(base, "/api/voice/settings", timeout=5)
    after = json.loads(raw)
    if after.get("fallback_reason"):
        failures.append("发生降级：%s" % after["fallback_reason"])
    else:
        print("实际后端=%s，无降级" % after["backend"])

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("VOICE_SMOKE=PASS（重启后第一句出声，%.1f 秒）" % elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
