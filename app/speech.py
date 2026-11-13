"""统一语音通道（Speech Policy Engine，第一批 2026-11-13）。

所有服务端喇叭播报（提醒/学习/坐姿/试听）都走这一条通道：

- 串行互斥：单 worker 线程逐句播 afplay，物理上杜绝叠话
  （旧架构 4 个调用点并发直调 voice.speak，会自己跟自己抢话）；
- 优先级：P0 显式提醒 > P1 交互应答 > P2 学习计时 > P3 坐姿警告 > P4 柔性确认；
- 场合感知：
  * 手动勿扰（文件式状态，跨进程共享——Agent 子进程工具也能设置）：
    P2-P4 直接压制转屏幕，P0 延后（提醒保持 active 由既有 15s 重试自然延后），
    P1 交互应答不受限（用户自己动作的回应）；
  * 系统静音/音量0（osascript，5 秒缓存）：一切播报都是"没送达"，
    P0 返回 muted 且**不标 delivered**（P0-3 根治：静音≠已送达）；
  * 安静时段（默认 22:00-07:00，可配置）：只压 P2-P4 自主播报，P0 照响、P1 照答；
- 保鲜期：过期不补播（学习类 120s、姿态类 60s），作废并记录；
- 屏幕兜底：被压制的话进 suppressed 环形缓冲（最近 5 条），
  /api/speech/status 透出给前端横幅——听不见也看得见；
- 送达回执：delivered / muted / deferred_dnd / suppress_quiet / suppress_dnd /
  stale / failed / queued —— 调用方按回执决定重试或落库。
"""
import collections
import heapq
import itertools
import json
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

PRIORITY_REMINDER = 0   # 用户显式提醒到点
PRIORITY_RESPONSE = 1   # 交互应答（试听等用户主动动作的回应），不受勿扰/安静时段限制
PRIORITY_STUDY = 2      # 学习定时（休息/回来/离座/回归）
PRIORITY_POSTURE = 3    # 坐姿警告
PRIORITY_GENTLE = 4     # 柔性确认（还有5分钟…）

REASON_LABELS = {
    'deferred_dnd': '勿扰中', 'suppress_dnd': '勿扰中', 'suppress_quiet': '安静时段',
    'muted': '系统静音', 'stale': '已过期', 'failed': '播放失败',
}


# ---------- 勿扰状态：文件式，跨进程（主服务 / MCP 子进程）共享 ----------

def _default_dnd_file() -> Path:
    from .config import settings
    return Path(settings.data_dir) / 'dnd.json'


def set_dnd(minutes: int = 60, path: Path = None) -> str:
    minutes = max(1, min(int(minutes), 8 * 60))
    until = (datetime.now().astimezone() + timedelta(minutes=minutes)).isoformat(timespec='seconds')
    p = Path(path or _default_dnd_file())
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'until': until}))
    tmp.replace(p)
    return until


def clear_dnd(path: Path = None):
    Path(path or _default_dnd_file()).unlink(missing_ok=True)


def dnd_until(path: Path = None) -> datetime | None:
    try:
        until = datetime.fromisoformat(json.loads(Path(path or _default_dnd_file()).read_text())['until'])
        return until if until > datetime.now().astimezone() else None
    except (OSError, ValueError, KeyError):
        return None


class _Item:
    __slots__ = ('text', 'priority', 'fresh_until', 'event', 'result', 'seq')

    def __init__(self, text, priority, fresh_until, event, seq):
        self.text, self.priority, self.fresh_until = text, priority, fresh_until
        self.event, self.result, self.seq = event, [None], seq

    def __lt__(self, other):
        return (self.priority, self.seq) < (other.priority, other.seq)


class SpeechChannel:
    def __init__(self, speak_fn, dnd_file: Path = None, quiet_start: int = 22, quiet_end: int = 7,
                 on_event=None):
        self._speak = speak_fn
        self._dnd_file = Path(dnd_file) if dnd_file else _default_dnd_file()
        self.quiet_start, self.quiet_end = quiet_start, quiet_end
        self._on_event = on_event
        self._heap: list[_Item] = []
        self._cv = threading.Condition()
        self._seq = itertools.count()
        self._stop = False
        self._thread = None
        self._suppressed = collections.deque(maxlen=5)
        self._mute_cache = (0.0, None)

    # ---------- 生命周期 ----------

    def start(self):
        if self._thread is None:
            self._stop = False
            self._thread = threading.Thread(target=self._worker, name='speech-channel', daemon=True)
            self._thread.start()

    def stop(self):
        self._stop = True
        with self._cv:
            self._cv.notify_all()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ---------- 场合感知 ----------

    def dnd_active(self) -> bool:
        return dnd_until(self._dnd_file) is not None

    def in_quiet(self, now: datetime = None) -> bool:
        now = now or datetime.now().astimezone()
        if self.quiet_start == self.quiet_end:
            return False
        if self.quiet_start < self.quiet_end:
            return self.quiet_start <= now.hour < self.quiet_end
        return now.hour >= self.quiet_start or now.hour < self.quiet_end  # 跨午夜

    def system_muted(self):
        """True/False/None(未知)。osascript 5 秒缓存；检测失败按未静音处理（不阻断播报）。"""
        mono, cached = self._mute_cache
        if cached is not None and time.monotonic() - mono < 5:
            return cached
        try:
            out = subprocess.run(
                ['osascript', '-e', 'get {output muted, output volume} of (get volume settings)'],
                capture_output=True, text=True, timeout=3)
            muted_s, vol_s = [s.strip() for s in out.stdout.strip().split(',')]
            val = muted_s == 'true' or int(vol_s) == 0
        except Exception:
            val = None
        self._mute_cache = (time.monotonic(), val)
        return val

    def _submit_policy(self, priority: int):
        """返回 None（可播）或压制原因。"""
        if priority == PRIORITY_RESPONSE:
            return 'muted' if self.system_muted() is True else None
        if priority == PRIORITY_REMINDER:
            if self.dnd_active():
                return 'deferred_dnd'
            if self.system_muted() is True:
                return 'muted'
            return None
        if self.dnd_active():
            return 'suppress_dnd'
        if self.in_quiet():
            return 'suppress_quiet'
        if self.system_muted() is True:
            return 'muted'
        return None

    def _record_suppressed(self, text, reason):
        last = self._suppressed[-1] if self._suppressed else None
        if last and last['text'] == text[:80] and last['reason'] == reason:
            return  # 同句同原因不重复记录（提醒 15s 重试会反复撞静音/勿扰，防刷事件）
        self._suppressed.append({'text': text[:80], 'reason': reason,
                                 'at': datetime.now().astimezone().isoformat(timespec='seconds')})
        if self._on_event:
            try:
                self._on_event({'type': 'speech_suppressed', 'reason': reason, 'text': text[:60]})
            except Exception:
                pass

    # ---------- 提交 ----------

    def submit(self, text: str, priority: int, fresh_seconds: int = None,
               wait: bool = False, timeout: float = 180) -> str:
        """提交一句话。wait=True 阻塞到播完（提醒循环用），否则入队即返回 'queued'。

        返回值即送达回执：delivered/queued/muted/deferred_dnd/suppress_dnd/
        suppress_quiet/stale/failed/timeout。
        """
        if not text or not text.strip():
            return 'failed'
        reason = self._submit_policy(priority)
        if reason:
            self._record_suppressed(text, reason)
            return reason
        fresh_until = None
        if fresh_seconds is not None:
            fresh_until = time.monotonic() + fresh_seconds
        item = _Item(text, priority, fresh_until,
                     threading.Event() if wait else None, next(self._seq))
        with self._cv:
            heapq.heappush(self._heap, item)
            self._cv.notify()
        if wait:
            if not item.event.wait(timeout):
                return 'timeout'
            return item.result[0]
        return 'queued'

    # ---------- worker ----------

    def _process_next(self, block: bool = True, timeout: float = 1.0) -> bool:
        """弹出下一句并播放。返回是否处理了一条。测试可直接驱动。"""
        with self._cv:
            if not self._heap:
                if not block:
                    return False
                self._cv.wait(timeout)
                if not self._heap:
                    return False
            item = heapq.heappop(self._heap)
        # 出队时二次策略检查（排队期间可能进了勿扰/安静时段/过期）
        if item.fresh_until is not None and time.monotonic() > item.fresh_until:
            item.result[0] = 'stale'
            self._record_suppressed(item.text, 'stale')
            if item.event:
                item.event.set()
            return True
        if item.priority >= PRIORITY_STUDY:
            reason = self._submit_policy(item.priority)
            if reason:
                item.result[0] = reason
                self._record_suppressed(item.text, reason)
                if item.event:
                    item.event.set()
                return True
        if self.system_muted() is True and item.priority in (PRIORITY_REMINDER, PRIORITY_RESPONSE):
            item.result[0] = 'muted'
            self._record_suppressed(item.text, 'muted')
            if item.event:
                item.event.set()
            return True
        try:
            played = self._speak(item.text)
            item.result[0] = 'delivered' if played == 'spoken' else 'failed'
        except Exception:
            item.result[0] = 'failed'
        if self._on_event:
            try:
                self._on_event({'type': 'speech_result', 'status': item.result[0], 'text': item.text[:60]})
            except Exception:
                pass
        if item.event:
            item.event.set()
        return True

    def _worker(self):
        while not self._stop:
            try:
                self._process_next(block=True, timeout=1.0)
            except Exception:
                time.sleep(0.5)

    # ---------- 状态 ----------

    def status(self) -> dict:
        until = dnd_until(self._dnd_file)
        return {
            'dnd_active': until is not None,
            'dnd_until': until.isoformat() if until else None,
            'quiet_now': self.in_quiet(),
            'quiet_hours': f"{self.quiet_start:02d}:00-{self.quiet_end:02d}:00",
            'muted': self.system_muted(),
            'queue_len': len(self._heap),
            'suppressed': [{'text': s['text'], 'reason': REASON_LABELS.get(s['reason'], s['reason']),
                            'at': s['at']} for s in self._suppressed],
        }
