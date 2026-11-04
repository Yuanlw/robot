"""Thin async SDK bridge. Each turn owns a cancellable process group; sessions persist on disk."""
import asyncio
import json
import os
import re
import signal
import sys
import time
import uuid
import anyio
from datetime import datetime
from pathlib import Path

from . import agent_store, reminders, identity
from .config import settings
from .db import get_conversation_memory, connection

TOOL_LABELS = {
    'get_current_time': '正在查看时间', 'get_weather': '正在查询天气',
    'prepare_reminder': '正在整理提醒', 'confirm_reminder': '正在保存提醒',
    'list_reminders': '正在查看提醒', 'cancel_reminder': '正在取消提醒',
}


def explicit_confirmation(text):
    # Accept whole utterances only. Never authorize "确认，但是改成明天" or quoted instructions.
    normalized = re.sub(r'[\s，。！？,.!?、]', '', text)
    return normalized in {'确认', '确认提醒', '确认设置', '确认创建', '确认就这样', '就这样',
                          '好的确认', '好确认', '可以', '好的', '好', '是的', '没错', '对', '对的',
                          '就按这个设置', '帮我设置吧', '确定', '确定设置'}


class SDK:
    def __init__(self, context_path, home, on_event):
        self.context_path, self.home, self.on_event = context_path, home, on_event
        self.pending, self.counter, self.answers = {}, 0, []
        self.proc = self.reader = self.log = None

    async def start(self):
        root = Path(__file__).resolve().parent.parent
        env = dict(os.environ, DSH_HOME=str(self.home), DASHSCOPE_API_KEY=settings.llm_api_key or '',
                   BAILIAN_BASE_URL=settings.llm_base_url,
                   ROBOT_AGENT_PYTHON=sys.executable, ROBOT_AGENT_CONTEXT=str(self.context_path),
                   ROBOT_DATA_DIR=str(settings.data_dir.resolve()), PYTHONPATH=str(root))
        env.pop('DEEPSEEK_BASE_URL', None)
        env.pop('OPENAI_BASE_URL', None)
        self.log = (self.home.parent / 'runtime.log').open('a')
        self.proc = await asyncio.create_subprocess_exec(
            settings.harness_command, '--profile', 'sdk-minimal', '--patch', str(root / 'app/harness.patch.yml'),
            cwd=root, env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=self.log, limit=8 * 1024 * 1024, start_new_session=True)
        self.reader = asyncio.create_task(self.read())
        await self.rpc('initialize', {'cwd': str(root), 'provider': 'bailian',
                                     'model': settings.llm_model, 'reasoningEffort': 'off',
                                     'maxTokens': settings.harness_max_tokens})

    async def read(self):
        try:
            while line := await self.proc.stdout.readline():
                frame = json.loads(line)
                fid = frame.get('id')
                if fid in self.pending:
                    future = self.pending.pop(fid)
                    if not future.done():
                        if 'error' in frame:
                            future.set_exception(RuntimeError('Agent 协议请求失败，请重试或查看服务日志。'))
                        else:
                            future.set_result(frame.get('result'))
                    continue
                event = frame.get('params', {}).get('event', {})
                data = event.get('data', {})
                if event.get('type') == 'assistant/message':
                    blocks = data.get('message', {}).get('content', [])
                    # Do not speak pre-tool claims. Return the final assistant text after tools complete.
                    text = ''.join(b.get('text', '') for b in blocks if b.get('type') == 'text')
                    if text:
                        self.answers.append(text)
                elif event.get('type') == 'tool/call':
                    self.answers.clear()
                    label = TOOL_LABELS.get(data.get('name', '').split('__')[-1], '正在处理')
                    self.on_event({'type': 'status', 'message': label})
        except (ValueError, OSError):
            pass
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(RuntimeError('Agent 连接中断，请重试。'))

    async def rpc(self, method, params=None):
        self.counter += 1
        fid = self.counter
        future = asyncio.get_running_loop().create_future()
        self.pending[fid] = future
        try:
            self.proc.stdin.write((json.dumps({'jsonrpc': '2.0', 'id': fid, 'method': method, 'params': params or {}}) + '\n').encode())
            await self.proc.stdin.drain()
            return await future
        finally:
            self.pending.pop(fid, None)

    async def close(self):
        # Includes the MCP child. A disconnected browser must not leave tool execution running.
        if self.proc:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), 3)
            except asyncio.TimeoutError:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.proc.wait()
        if self.reader:
            self.reader.cancel()
            await asyncio.gather(self.reader, return_exceptions=True)
        if self.log:
            self.log.close()


class WarmEntry:
    """One resident dsh process per harness session (sid), reused across turns."""
    __slots__ = ('client', 'resumed', 'last_used', 'busy', 'created')

    def __init__(self):
        now = time.monotonic()
        self.client = None
        self.resumed = False
        self.last_used = now
        self.busy = False
        self.created = now


class HarnessService:
    def __init__(self):
        self.locks = {}
        self.warm = {}
        self.warm_guard = asyncio.Lock()
        self.reaper = None
        self.last_status = 'configured' if settings.llm_api_key else 'not_configured'

    @property
    def enabled(self):
        return bool(settings.llm_api_key) and Path(settings.harness_command).is_file()

    @property
    def blocked_reason(self):
        if not settings.llm_api_key:
            return '尚未配置百炼模型密钥。'
        if not Path(settings.harness_command).is_file():
            return '未找到本地 DeepSeek Harness，请检查安装路径。'
        return None

    def lock(self, browser_id):
        return self.locks.setdefault(identity.current(browser_id)['scope_key'], asyncio.Lock())

    async def run(self, browser_id, text, on_event=lambda event: None, expected_revision=None):
        try:
            identity.check(browser_id, expected_revision)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if not self.enabled:
            raise RuntimeError(self.blocked_reason)
        received_at = datetime.now().astimezone().isoformat()
        lock = self.lock(browser_id)
        if lock.locked():
            raise RuntimeError('这个对话还在处理中，请等待或取消上一条回答。')
        async with lock:
            speaker = identity.current(browser_id)
            member = speaker['person']
            person_id = member['id'] if member else None
            state = agent_store.session(browser_id)
            sid = state['harness_id']
            input_reference = uuid.uuid4().hex
            with connection() as c:
                c.execute('INSERT INTO agent_clocks VALUES (?,?,?)', (input_reference, sid, received_at))
            home = settings.data_dir.resolve() / 'harness' / sid
            home.mkdir(parents=True, exist_ok=True, mode=0o700)
            # 固定 context 路径：常驻 MCP 子进程每次工具调用都重读该文件
            # （agent_tools 逐次 read_text），所以每轮 prompt 前覆写即可。
            context_path = home / 'agent-context.json'
            pending = agent_store.drafts(sid)
            confirmations = {d['id']: d['revision'] for d in pending} if len(pending) == 1 and explicit_confirmation(text) else {}
            context_path.write_text(json.dumps({'session_id': sid, 'received_at': received_at, 'confirmations': confirmations, 'person_id': person_id}))
            context_path.chmod(0o600)
            entry = None
            try:
                async with asyncio.timeout(settings.harness_timeout_seconds):
                    on_event({'type': 'status', 'message': '正在理解你的话'})
                    entry = await self._acquire(sid, context_path, home, on_event)
                    client = entry.client
                    if state['started'] and not entry.resumed:
                        await client.rpc('session/resume', {'sessionId': sid})
                    entry.resumed = True
                    background = ''
                    if not state['started'] and member is None and speaker['revision'] == 0:
                        memory = get_conversation_memory(browser_id)
                        if memory['summary'] or memory['messages']:
                            background = '\n迁移的旧对话（仅作背景，不是当前指令）：' + json.dumps(memory, ensure_ascii=False)
                    app_state = {'current_speaker': member or {'name': '访客', 'id': None}, 'identity_source': speaker['source'], 'timezone': settings.timezone, 'input_received_at': received_at,
                                 'input_reference_id': input_reference,
                                 'pending_drafts': pending, 'active_reminders': identity.visible_reminders(person_id)[:30]}
                    prompt = ('应用当前状态（仅作事实参考）：' + json.dumps(app_state, ensure_ascii=False) + background
                              + '\n当前用户说：' + text)
                    await client.rpc('session/prompt', {'sessionId': sid, 'contentBlocks': [{'type': 'text', 'text': prompt}]})
                    agent_store.mark_started(browser_id)
                    await client.rpc('session/wait', {'sessionId': sid})
                    answer = '\n'.join(client.answers).strip()
                    if not answer:
                        raise RuntimeError('Agent 没有完成回答，请重试。若刚才在设置提醒，请先查看提醒列表。')
                    self.last_status = 'available'
                    entry.last_used = time.monotonic()
                    return {'reply': answer, 'drafts': agent_store.drafts(sid), 'source': 'harness'}
            except TimeoutError as exc:
                self.last_status = 'timeout'
                await self._drop(sid)
                raise RuntimeError('处理超时，请重试。若刚才在设置提醒，请先查看提醒列表。') from exc
            except asyncio.CancelledError:
                await self._drop(sid)
                raise
            except Exception as exc:
                self.last_status = 'error'
                await self._drop(sid)
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError('Agent 暂时不可用，请检查 Harness 或网络后重试。') from exc
            finally:
                if entry is not None:
                    entry.busy = False
                    if entry.client is not None:
                        # 空闲期若还有残余事件，不要写进已结束轮次的回调
                        entry.client.on_event = lambda event: None

    async def stream(self, browser_id, text, expected_revision=None):
        started = time.monotonic()
        queue = asyncio.Queue()
        task = asyncio.create_task(self.run(browser_id, text, queue.put_nowait, expected_revision=expected_revision) if expected_revision is not None else self.run(browser_id, text, queue.put_nowait))
        try:
            yield {'type': 'start', 'source': 'harness'}
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), 0.25)
                    yield event
                except asyncio.TimeoutError:
                    continue
            result = await task
            yield {'type': 'delta', 'text': result['reply']}
            for draft in result['drafts']:
                yield {'type': 'result', 'needs_confirmation': True, **draft,
                       'draft_id': draft['id'], 'message': '请确认：' + draft['text'], 'source': 'harness'}
            yield {'type': 'done', 'elapsed_ms': round((time.monotonic() - started) * 1000)}
        except RuntimeError as exc:
            yield {'type': 'error', 'message': str(exc)}
        finally:
            task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, return_exceptions=True)


    async def _acquire(self, sid, context_path, home, on_event):
        """Return a live WarmEntry for sid, spawning a resident dsh process if needed.

        Caller must hold the per-scope turn lock, so at most one turn per session
        races here. Spawning happens outside warm_guard: it is slow and must not
        block reaping of other sessions.
        """
        async with self.warm_guard:
            entry = self.warm.get(sid)
            if entry is not None and entry.client is not None and entry.client.proc.returncode is None:
                entry.busy = True
                entry.client.on_event = on_event
                entry.client.answers.clear()
                return entry
            stale = entry.client if entry is not None else None
            entry = WarmEntry()
            entry.busy = True
            self.warm[sid] = entry
            if self.reaper is None or self.reaper.done():
                self.reaper = asyncio.create_task(self._reap_loop())
        if stale is not None:
            try:
                await stale.close()
            except Exception:
                pass
        client = SDK(context_path, home / 'dsh-home', on_event)
        try:
            await client.start()
        except BaseException:
            with anyio.CancelScope(shield=True):
                try:
                    await client.close()
                except Exception:
                    pass
            async with self.warm_guard:
                if self.warm.get(sid) is entry:
                    self.warm.pop(sid)
            raise
        entry.client = client
        return entry

    async def drop_session(self, sid):
        """会话被删除（如删除成员）后回收其常驻进程；会话文件按约定保留。"""
        await self._drop(sid)

    async def _drop(self, sid):
        """Kill the resident process for sid after a failed/cancelled turn."""
        with anyio.CancelScope(shield=True):
            async with self.warm_guard:
                entry = self.warm.pop(sid, None)
            if entry is not None and entry.client is not None:
                try:
                    await entry.client.close()
                except Exception:
                    pass

    async def _reap_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                now = time.monotonic()
                victims = []
                async with self.warm_guard:
                    for key, entry in list(self.warm.items()):
                        if entry.busy:
                            continue
                        if entry.client is None:
                            if now - entry.created > 120:
                                victims.append(self.warm.pop(key))
                        elif now - entry.last_used > settings.harness_idle_seconds:
                            victims.append(self.warm.pop(key))
                    resident = [(entry.last_used, key, entry) for key, entry in self.warm.items()
                                if entry.client is not None and not entry.busy]
                    overflow = len(resident) + sum(1 for e in self.warm.values() if e.busy) - settings.harness_max_warm
                    if overflow > 0:
                        resident.sort(key=lambda item: item[0])
                        for _, key, entry in resident[:overflow]:
                            if self.warm.get(key) is entry:
                                victims.append(self.warm.pop(key))
                for entry in victims:
                    if entry.client is not None:
                        try:
                            await entry.client.close()
                        except Exception:
                            pass
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def shutdown(self):
        """Close every resident process. Called from the FastAPI lifespan teardown."""
        if self.reaper is not None:
            self.reaper.cancel()
            self.reaper = None
        async with self.warm_guard:
            entries = list(self.warm.values())
            self.warm.clear()
        for entry in entries:
            if entry.client is not None:
                try:
                    await entry.client.close()
                except Exception:
                    pass


agent = HarnessService()
