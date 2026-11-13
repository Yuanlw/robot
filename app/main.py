from datetime import datetime
from pathlib import Path
from typing import Literal
from contextlib import asynccontextmanager
import asyncio
import io
import json
import time
import threading
import tempfile
import wave

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .db import init_db, list_events, record_event, get_conversation_memory, save_conversation_memory, delete_conversation_memory
from .commands import parse_reminder
from .face import FaceService
from .reminders import cancel, create, due_now, list_active, mark_delivered, update
from .voice import VoiceService
from .llm import LLMService
from .config import settings
from .harness import agent
from . import agent_store, bailian, identity, study, study_profiler
from . import study_report
from . import vision as vision_mod
from . import speech
from .csrf import CSRFMiddleware

faces = FaceService()
voice = VoiceService()
llm = LLMService()
vision = vision_mod.VisionService()
speech_channel = speech.SpeechChannel(
    lambda text: voice.speak(text),
    quiet_start=settings.quiet_start_hour, quiet_end=settings.quiet_end_hour,
    on_event=lambda ev: record_event(ev.pop('type'), ev))
conversation_histories: dict[str, dict[str, object]] = {}
conversation_lock = threading.Lock()
RECENT_CONVERSATION_MESSAGES = 8
MAX_CONVERSATION_MESSAGES = 20


async def reminder_loop():
    last_suppressed: dict = {}
    while True:
        try:
            reminders = due_now()
        except Exception:
            reminders = []
        for reminder in reminders:
            rid = reminder["id"]
            try:
                person = f"{reminder['person_name']}，" if reminder.get("person_name") else ""
                spoken = f"{person}提醒时间到了：{reminder['text']}"
                # 统一语音通道：只有真播出去（delivered）才标送达；
                # 静音/勿扰返回回执但不标——提醒保持 active，15s 循环自然延后重试。
                result = await asyncio.to_thread(speech_channel.submit, spoken,
                                                 speech.PRIORITY_REMINDER, None, True, 180)
                if result == "delivered":
                    last_suppressed.pop(rid, None)
                    mark_delivered(rid)
                    record_event("reminder_delivered", {"reminder_id": rid, "text": spoken})
                elif result in ("muted", "deferred_dnd"):
                    if last_suppressed.get(rid) != result:  # 状态变化才记录，防 15s 刷屏
                        last_suppressed[rid] = result
                        record_event("reminder_suppressed", {"reminder_id": rid, "reason": result})
                else:
                    raise RuntimeError(f"提醒未播放（{result}），保留待提醒状态。")
            except Exception as exc:
                record_event("reminder_error", {"reminder_id": rid, "error": str(exc)[:120]})
        await asyncio.sleep(15)


async def study_loop():
    """学习陪伴循环（统一语音通道版 2026-11-13）。

    每 30 秒：① 会话健康收口（超龄/闲置自动结束，视觉断连暂停、恢复续跑）
    ② 定时提醒判定 → 提交通道（勿扰/安静时段/静音由通道策略层统一处理，
    新鲜期 120 秒——睡眠唤醒后不补播陈旧话术）。
    """
    if not settings.study_enabled:
        return
    while True:
        try:
            now = datetime.now().astimezone()

            for sid, pid, action, reason in study.reconcile_sessions(now):
                vision_mod.judge.reset(pid)
                record_event("study_auto", {"session_id": sid, "action": action, "reason": reason})

            def resolver(person_id):
                try:
                    return study_profiler.thresholds_for_reminder(person_id)
                except Exception:
                    return None

            needs = study.sessions_needing_reminder(
                settings.study_focus_minutes, settings.study_break_minutes, threshold_resolver=resolver)
            for item in needs:
                sess = item['session']
                reminder_type = item['type']
                spoken = item['spoken']
                if reminder_type == 'break_time':
                    study.pause_session(sess['id'])
                elif reminder_type == 'resume_time':
                    study.resume_session(sess['id'])
                prio = speech.PRIORITY_GENTLE if reminder_type == 'gentle' else speech.PRIORITY_STUDY
                res = speech_channel.submit(spoken, prio, fresh_seconds=120)
                study.record_reminder(sess['id'], reminder_type, str(item['elapsed']['phase_elapsed_seconds'] // 60))
                record_event("study_reminder", {"session_id": sess['id'], "type": reminder_type,
                                                "text": spoken, "speech": res})

        except Exception as e:
            record_event("study_loop_error", {"error": str(e)})
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    agent_store.init_agent_db()
    study.init_study_db()
    vision_mod.init_vision_db()
    faces.init_face_db()
    await asyncio.to_thread(voice.warmup)
    speech_channel.start()
    reminder_task = asyncio.create_task(reminder_loop())
    study_task = asyncio.create_task(study_loop())
    try:
        yield
    finally:
        reminder_task.cancel()
        study_task.cancel()
        await asyncio.gather(reminder_task, study_task, return_exceptions=True)
        speech_channel.stop()
        await agent.shutdown()


app = FastAPI(title="家庭机器人", version="0.4.0", lifespan=lifespan)
app.add_middleware(CSRFMiddleware)


class ReminderIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)
    due_at: datetime
    person_id: int | None = None
    repeat_rule: Literal["once"] = "once"
    scope: Literal["personal", "household"] | None = None

    @field_validator('due_at')
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError('提醒时间必须带时区')
        return value

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("提醒内容不能为空")
        return value


@app.get("/", response_class=HTMLResponse)
def home():
    return (Path(__file__).with_name("web.html")).read_text()


@app.get("/static/robot-avatar.svg")
def robot_avatar():
    return FileResponse(Path(__file__).parent.parent / "docs" / "产品设计-v1" / "robot-avatar.svg", media_type="image/svg+xml")


@app.get("/static/worklet.js")
def worklet_js():
    # 麦克风 AudioWorklet：采集 → 重采样 16k Int16 → 电平上报（移植自 voice-poc demo）
    return Response((Path(__file__).with_name("worklet.js")).read_text(),
                    media_type="application/javascript")


@app.get("/api/health")
def health():
    backend = agent if settings.agent_backend == 'harness' else llm
    return {"ok": True, "service": "family-robot", "version": app.version, "agent_backend": settings.agent_backend, "llm_enabled": backend.enabled, "llm_blocked": backend.blocked_reason, "voice": voice.status(), "voice_details": voice.details(), "stt": settings.stt_backend, "llm_model": settings.llm_model, "llm_status": backend.last_status}


class VoicePreferenceIn(BaseModel):
    voice: str = Field(min_length=1, max_length=32)


@app.get("/api/voice/settings")
def voice_settings():
    return voice.details()


@app.put("/api/voice/settings")
def update_voice_settings(payload: VoicePreferenceIn):
    try:
        voice.set_voice(payload.voice)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    return voice.details()


@app.get("/api/reminders")
def reminders(session_id: str | None = None):
    if session_id is None:
        return list_active()
    member = identity.current(session_id)['person']
    return identity.visible_reminders(member['id'] if member else None)


@app.get("/api/events")
def events(session_id: str | None = None):
    if session_id is None:
        return list_events()
    member = identity.current(session_id)['person']
    result = []
    for event in list_events():
        rid = event['payload'].get('reminder_id')
        if rid:
            try:
                identity.require_reminder(rid, member['id'] if member else None)
                result.append(event)
            except ValueError:
                pass
    return result


@app.post("/api/reminders")
def add_reminder(payload: ReminderIn, session_id: str | None = None, identity_revision: int | None = None):
    if session_id is not None:
        state = check_identity(session_id, identity_revision)
        member = state['person']
        payload.person_id = member['id'] if member and payload.scope != 'household' else None
        if payload.scope == 'personal' and member is None:
            raise HTTPException(409, '请先选择家庭成员')
    if payload.person_id is not None and not identity.person(payload.person_id):
        raise HTTPException(422, '家庭成员不存在')
    if payload.due_at <= datetime.now().astimezone():
        raise HTTPException(422, "提醒时间需要设为未来时间。")
    return create(payload.text, payload.due_at, payload.person_id, payload.repeat_rule)


@app.patch("/api/reminders/{reminder_id}")
def edit_reminder(reminder_id: int, payload: ReminderIn, session_id: str | None = None, identity_revision: int | None = None):
    if session_id is not None:
        check_reminder_owner(reminder_id, session_id, identity_revision)
    if payload.due_at <= datetime.now().astimezone():
        raise HTTPException(422, "提醒时间需要设为未来时间。")
    item = update(reminder_id, payload.text, payload.due_at)
    if not item:
        raise HTTPException(404, "提醒不存在或已经送达/取消。")
    return item


@app.delete("/api/reminders/{reminder_id}")
def delete_reminder(reminder_id: int, session_id: str | None = None, identity_revision: int | None = None):
    if session_id is not None:
        check_reminder_owner(reminder_id, session_id, identity_revision)
    if not cancel(reminder_id):
        raise HTTPException(404, "提醒不存在或已取消")
    return {"ok": True}


def check_identity(session_id, revision=None):
    try:
        return identity.check(session_id, revision)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


def check_reminder_owner(reminder_id, session_id, revision=None):
    state = check_identity(session_id, revision)
    try:
        return identity.require_reminder(reminder_id, state['person']['id'] if state['person'] else None)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


class SpeakerIn(BaseModel):
    person_id: int | None = Field(default=None, ge=1)
    revision: int = Field(ge=0)


@app.get('/api/people')
def list_people():
    people = identity.people()
    counts = faces.sample_counts()
    for p in people:
        p['samples'] = counts.get(p['id'], 0)
    return people


@app.delete('/api/people/{person_id}')
async def delete_person(person_id: int):
    # 与人脸识别同样的保护：正在回答时删除会让进行中的一轮失去归属
    lock = agent.locks.get('person:' + str(person_id))
    if lock is not None and lock.locked():
        raise HTTPException(409, '该成员正在对话中，请先结束当前回答再删除。')
    try:
        harness_id = identity.delete_person(person_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if harness_id:
        await agent.drop_session(harness_id)
    record_event('person_deleted', {'person_id': person_id})
    return {'deleted': person_id}


@app.get('/api/conversations/{session_id}/identity')
def current_speaker(session_id: str):
    return identity.public(session_id)


def select_speaker(session_id, person_id, revision, source='manual'):
    if agent.lock(session_id).locked():
        raise HTTPException(409, '请先结束当前回答或识别，再切换成员')
    if person_id is not None:
        target_lock = agent.locks.get('person:' + str(person_id))
        if target_lock and target_lock.locked():
            raise HTTPException(409, '该成员的对话正在处理中，请稍后再切换')
    try:
        return identity.select(session_id, person_id, source, revision)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.put('/api/conversations/{session_id}/identity')
async def change_speaker(session_id: str, payload: SpeakerIn):
    return select_speaker(session_id, payload.person_id, payload.revision)


async def face_image(image):
    content = await image.read(10 * 1024 * 1024 + 1)
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, '照片不能超过10MB')
    with tempfile.NamedTemporaryFile(prefix='robot-face-', suffix='.jpg', delete=False) as file:
        file.write(content)
        return Path(file.name)


@app.post("/api/faces/enroll")
async def enroll_face(name: str = Form(..., min_length=1, max_length=80), append: bool = Form(False),
                      image: UploadFile = File(...)):
    if not name.strip():
        raise HTTPException(422, '请填写姓名')
    path = await face_image(image)
    try:
        return await asyncio.to_thread(faces.enroll, name.strip(), str(path), append)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/faces/recognize-burst")
async def recognize_face_burst(images: list[UploadFile] = File(...), session_id: str | None = Form(None),
                               identity_revision: int | None = Form(None)):
    """多帧投票识别（3 帧至少 2 帧一致才认定），协议与 /api/faces/recognize 一致。"""
    if not 1 <= len(images) <= 5:
        raise HTTPException(422, '请提交 1-5 帧画面')
    if session_id:
        check_identity(session_id, identity_revision)
        if agent.lock(session_id).locked():
            raise HTTPException(409, '请先结束当前回答，再识别成员')
    paths = [await face_image(im) for im in images]
    try:
        before = check_identity(session_id, identity_revision)['revision'] if session_id else None
        result = await asyncio.to_thread(faces.recognize_burst, [str(p) for p in paths])
        if session_id:
            found = result['faces']
            person_id = found[0].get('person_id') if len(found) == 1 else None
            result['identity'] = select_speaker(session_id, person_id, before, 'face')
            result['needs_selection'] = person_id is None
            result['message'] = (
                f"认出你了，{result['identity']['person']['name']}（{result.get('matched_frames', 1)}/{result.get('frames', 1)} 帧一致）。现在可以继续聊天。"
                if person_id else
                ("画面里有多个人，请单独正对摄像头识别，或手动选择成员。"
                 if result.get('multi_face_frames') else
                 f"还不能确认你是谁（{result.get('matched_frames', 0)}/{result.get('frames', 0)} 帧一致），请选择成员、重新录入或继续访客聊天。"))
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        for p in paths:
            p.unlink(missing_ok=True)


@app.get("/api/faces/recog-stats")
async def face_recog_stats(days: int = 7):
    """识别距离分布统计（E：阈值数据驱动）。仅本机使用，不做鉴权隔离。"""
    return faces.stats(days)


@app.post("/api/faces/recognize")
async def recognize_face(image: UploadFile = File(...), session_id: str | None = Form(None), identity_revision: int | None = Form(None)):
    if session_id:
        check_identity(session_id, identity_revision)
        if agent.lock(session_id).locked():
            raise HTTPException(409, '请先结束当前回答，再识别成员')
    path = await face_image(image)
    try:
        # Recognition is read-only until it has finished; revision detects any intervening switch.
        before = check_identity(session_id, identity_revision)['revision'] if session_id else None
        result = await asyncio.to_thread(faces.recognize, str(path))
        if session_id:
            found = result['faces']
            person_id = found[0].get('person_id') if len(found) == 1 else None
            result['identity'] = select_speaker(session_id, person_id, before, 'face')
            result['needs_selection'] = person_id is None
            result['message'] = ('认出你了，' + result['identity']['person']['name'] + '。现在可以继续聊天。') if person_id else (
                '画面中有多人，请选择当前说话人。' if len(found) > 1 else '还不能确认你是谁，请选择成员或继续访客聊天。')
        return result
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/speak")
def speak(text: str):
    return {"text": text, "status": speech_channel.submit(text, speech.PRIORITY_RESPONSE, wait=True, timeout=60)}


@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    """浏览器 AudioWorklet 录音（16bit 单声道 WAV）→ qwen3-asr-flash 整句识别。

    静默门控必须在送识别之前做：LLM 架构的 ASR 喂全零静音不会返回空，
    而是幻觉出文本（demo 2026-11-03 实测）。
    """
    if settings.stt_backend != 'dashscope':
        raise HTTPException(501, '服务端语音识别未启用，请改用浏览器语音识别。')
    data = await audio.read()
    if len(data) > 12 * 1024 * 1024:
        raise HTTPException(413, '录音过长，请分段后再试。')
    try:
        with wave.open(io.BytesIO(data)) as wav_file:
            channels, width = wav_file.getnchannels(), wav_file.getsampwidth()
            rate = wav_file.getframerate()
            pcm = wav_file.readframes(wav_file.getnframes())
    except Exception as exc:
        raise HTTPException(422, '无法解析录音，请上传 WAV 音频。') from exc
    if channels != 1 or width != 2:
        raise HTTPException(422, '仅支持 16bit 单声道 WAV 录音。')
    silent, stats = bailian.is_silence(pcm, rate)
    if silent:
        return {'text': '', 'silent': True, 'stt_ms': 0, 'stats': stats}
    started = time.monotonic()
    try:
        text = await asyncio.to_thread(bailian.stt_transcribe, pcm, rate)
    except bailian.BailianError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {'text': text, 'silent': False, 'stats': stats,
            'stt_ms': round((time.monotonic() - started) * 1000)}


@app.post("/api/command")
async def command(text: str = Query(min_length=1, max_length=2000), session_id: str = Query(default='default', min_length=1, max_length=100), identity_revision: int | None = None):
    if settings.agent_backend == 'harness':
        try:
            return await agent.run(session_id, text, expected_revision=identity_revision) if identity_revision is not None else await agent.run(session_id, text)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc
    return await asyncio.to_thread(legacy_command, text)


def legacy_command(text: str):
    try:
        parsed = parse_reminder(text)
    except ValueError as exc:
        raise HTTPException(400, "时间格式不正确，请使用有效的小时和分钟。") from exc
    if parsed:
        reminder_text, due_at = parsed
        return {"needs_confirmation": True, "text": reminder_text, "due_at": due_at.isoformat(), "message": f"确认一下，{due_at:%m月%d日%H点%M分}提醒你{reminder_text}，可以吗？", "source": "local"}
    if llm.enabled:
        try:
            return {"needs_confirmation": False, "reply": llm.chat(text), "source": "llm"}
        except RuntimeError as exc:
            if llm.blocked_reason:
                return {"needs_confirmation": False, "reply": "我现在可以设置提醒，但聊天模型暂时不可用。请先使用‘提醒我明天晚上八点练琴’这类指令。", "source": "local-fallback", "detail": str(exc)}
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, f"模型调用失败：{exc}") from exc
    if llm.blocked_reason:
        return {"needs_confirmation": False, "reply": "我现在可以设置提醒，但聊天模型暂时不可用。请先使用‘提醒我明天晚上八点练琴’这类指令。", "source": "local-fallback", "detail": llm.blocked_reason}
    raise HTTPException(400, "目前支持的格式：提醒我明天晚上八点练琴；复杂表达需要配置模型")


@app.post("/api/chat")
async def chat(text: str = Query(min_length=1, max_length=2000), session_id: str = Query(default='default', min_length=1, max_length=100), identity_revision: int | None = None):
    if settings.agent_backend == 'harness':
        return await command(text, session_id, identity_revision)
    return await asyncio.to_thread(legacy_chat, text)


def legacy_chat(text: str):
    if not llm.enabled:
        raise HTTPException(503, "LLM 未配置")
    try:
        return {"reply": llm.chat(text), "model": settings.llm_model}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"模型调用失败：{exc}") from exc


class SpeechIn(BaseModel):
    text: str = Field(min_length=1, max_length=1500)


@app.post("/api/voice/audio")
def voice_audio(payload: SpeechIn):
    try:
        audio = voice.synthesize(payload.text)
        return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store", "X-Voice-Backend": voice.status()})
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "语音暂时不可用，请检查中文语音配置。") from exc


@app.post("/api/command/stream")
async def stream_command(text: str = Query(min_length=1, max_length=2000), session_id: str = Query(default="default", min_length=1, max_length=100), identity_revision: int | None = None):
    if settings.agent_backend == 'harness':
        async def stream():
            async for event in agent.stream(session_id, text, expected_revision=identity_revision):
                yield 'data: ' + json.dumps(event, ensure_ascii=False) + '\n\n'
        return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    return legacy_stream_command(text, session_id)


def legacy_stream_command(text, session_id):
    try:
        parsed = parse_reminder(text)
    except ValueError as exc:
        raise HTTPException(400, "时间格式不正确，请使用有效的小时和分钟。") from exc

    def emit(data):
        return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

    def events():
        started = time.monotonic()
        yield emit({"type": "start"})
        if parsed:
            reminder_text, due_at = parsed
            yield emit({"type": "result", "needs_confirmation": True,
                        "text": reminder_text, "due_at": due_at.isoformat(),
                        "message": f"确认一下，{due_at:%m月%d日%H点%M分}提醒你{reminder_text}，可以吗？", "source": "local"})
        else:
            try:
                with conversation_lock:
                    memory = conversation_histories.get(session_id)
                if memory is None:
                    memory = get_conversation_memory(session_id)
                    with conversation_lock:
                        conversation_histories[session_id] = memory
                summary = str(memory["summary"])
                history = list(memory["messages"])
                if len(history) >= MAX_CONVERSATION_MESSAGES:
                    compact_count = len(history) - RECENT_CONVERSATION_MESSAGES
                    compact = history[:compact_count]
                    try:
                        summary = llm.summarize(summary, compact)
                    except RuntimeError as exc:
                        yield emit({"type": "error", "message": str(exc)})
                        return
                    history = history[compact_count:]
                    with conversation_lock:
                        conversation_histories[session_id] = {"summary": summary, "messages": list(history)}
                    save_conversation_memory(session_id, summary, history)
                messages = ([{"role": "system", "content": "此前较早对话的记忆摘要（作为背景事实使用；若用户后来更正，以最新说法为准）：" + summary}] if summary else []) + history + [{"role": "user", "content": text}]
                answer_parts = []
                for delta in llm.stream_messages(messages):
                    answer_parts.append(delta)
                    yield emit({"type": "delta", "text": delta})
                answer = "".join(answer_parts).strip()
                if answer:
                    with conversation_lock:
                        memory = conversation_histories.setdefault(session_id, {"summary": summary, "messages": []})
                        updated = memory["messages"]
                        updated.extend([{"role": "user", "content": text}, {"role": "assistant", "content": answer}])
                        saved_summary = str(memory["summary"])
                    save_conversation_memory(session_id, saved_summary, updated)
            except RuntimeError as exc:
                yield emit({"type": "error", "message": str(exc)})
                return
        yield emit({"type": "done", "elapsed_ms": round((time.monotonic() - started) * 1000)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/api/conversations/{session_id}")
async def clear_conversation(session_id: str, identity_revision: int | None = None):
    check_identity(session_id, identity_revision)
    if agent.lock(session_id).locked():
        raise HTTPException(409, '请先取消当前回答，再开启新对话。')
    async with agent.lock(session_id):
        agent_store.clear(session_id)
        with conversation_lock:
            conversation_histories.pop(session_id, None)
        delete_conversation_memory(session_id)
    return {"ok": True}


class DraftConfirmation(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=1)
    identity_revision: int | None = None


class DraftEdit(DraftConfirmation):
    text: str = Field(min_length=1, max_length=300)
    due_at: datetime


@app.get('/api/conversations/{session_id}/drafts')
def pending_drafts(session_id: str):
    return agent_store.drafts(agent_store.session(session_id)['harness_id'])


@app.post('/api/reminder-drafts/{draft_id}/confirm')
async def confirm_draft(draft_id: str, payload: DraftConfirmation):
    check_identity(payload.session_id, payload.identity_revision)
    if agent.lock(payload.session_id).locked():
        raise HTTPException(409, '对话正在处理，请稍后确认最新草稿。')
    async with agent.lock(payload.session_id):
        try:
            sid = agent_store.session(payload.session_id)['harness_id']
            return agent_store.confirm(sid, draft_id, payload.revision)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc


@app.patch('/api/reminder-drafts/{draft_id}')
async def edit_draft(draft_id: str, payload: DraftEdit):
    speaker = check_identity(payload.session_id, payload.identity_revision)
    if agent.lock(payload.session_id).locked():
        raise HTTPException(409, '对话正在处理，请稍后修改。')
    async with agent.lock(payload.session_id):
        sid = agent_store.session(payload.session_id)['harness_id']
        current = next((d for d in agent_store.drafts(sid) if d['id'] == draft_id), None)
        if not current or current['revision'] != payload.revision:
            raise HTTPException(409, '草稿已修改，请确认最新内容。')
        try:
            return agent_store.prepare(sid, {'draft_id': draft_id, 'text': payload.text, 'due_at': payload.due_at.isoformat()}, datetime.now().astimezone().isoformat(), speaker['person']['id'] if speaker['person'] else None)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


# ===== 学习陪伴 API =====

@app.post("/api/study/start")
async def study_start(session_id: str, identity_revision: int | None = None):
    """开始学习会话。"""
    if not settings.study_enabled:
        raise HTTPException(503, "学习陪伴功能未启用")
    speaker = check_identity(session_id, identity_revision)
    if not speaker['person']:
        raise HTTPException(400, "需要先选择身份")
    try:
        sess = study.start_session(speaker['person']['id'])
        record_event("study_start", {"session_id": sess['id'], "person_id": speaker['person']['id']})
        return {"ok": True, "session": sess}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/study/current")
async def study_current(session_id: str, identity_revision: int | None = None):
    """获取当前学习状态。"""
    if not settings.study_enabled:
        return {"enabled": False}
    speaker = check_identity(session_id, identity_revision)
    if not speaker['person']:
        return {"enabled": True, "session": None}
    sess = study.get_current_session(speaker['person']['id'])
    if not sess:
        return {"enabled": True, "session": None}
    elapsed = study.compute_elapsed(sess)
    return {"enabled": True, "session": sess, "elapsed": elapsed}


@app.post("/api/study/end")
async def study_end(session_id: str, identity_revision: int | None = None):
    """结束学习会话。"""
    if not settings.study_enabled:
        raise HTTPException(503, "学习陪伴功能未启用")
    speaker = check_identity(session_id, identity_revision)
    if not speaker['person']:
        raise HTTPException(400, "需要先选择身份")
    sess = study.get_current_session(speaker['person']['id'])
    if not sess:
        raise HTTPException(400, "当前没有进行中的学习会话")
    elapsed = study.compute_elapsed(sess)
    study.end_session(sess['id'])
    study_min = elapsed['study_seconds'] // 60
    break_min = elapsed['break_seconds'] // 60
    summary = f"今天学了 {study_min} 分钟"
    if break_min > 0:
        summary += f"，休息了 {break_min} 分钟"
    summary += "，辛苦了！"
    record_event("study_end", {"session_id": sess['id'], "elapsed": elapsed, "summary": summary})
    return {"ok": True, "elapsed": elapsed, "summary": summary}


class StudyRateIn(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    score: int = Field(ge=1, le=10)
    tags: list[str] = Field(default_factory=list, max_length=8)
    identity_revision: int | None = None


@app.get("/api/study/profile")
async def study_profile(session_id: str, identity_revision: int | None = None):
    """当前成员的学习画像（第一期：数据积累不足时返回默认节奏）。"""
    if not settings.study_enabled:
        return {"enabled": False}
    speaker = check_identity(session_id, identity_revision)
    if not speaker['person']:
        raise HTTPException(400, "需要先选择身份")
    return {"enabled": True, "profile": study_profiler.build_profile(speaker['person']['id'])}


@app.get("/api/study/report")
async def study_report_api(session_id: str, period: Literal["today", "week", "month"] = "week",
                           identity_revision: int | None = None):
    """学习报告（Markdown + 结构化数据）。"""
    if not settings.study_enabled:
        raise HTTPException(503, "学习陪伴功能未启用")
    speaker = check_identity(session_id, identity_revision)
    if not speaker['person']:
        raise HTTPException(400, "需要先选择身份")
    report = study_report.build_report(speaker['person']['id'], speaker['person']['name'], period)
    record_event("study_report", {"person_id": speaker['person']['id'], "period": period})
    return {"enabled": True, **report}


@app.post("/api/study/rate")
async def study_rate(payload: StudyRateIn):
    """自评本次学习质量（1-10 + 场景标签）。"""
    if not settings.study_enabled:
        raise HTTPException(503, "学习陪伴功能未启用")
    speaker = check_identity(payload.session_id, payload.identity_revision)
    if not speaker['person']:
        raise HTTPException(400, "需要先选择身份")
    sess = study.get_rateable_session(speaker['person']['id'])
    if not sess:
        raise HTTPException(400, "没有可评分的学习会话（进行中或刚结束30分钟内）")
    return study.rate_session(sess['id'], payload.score, payload.tags)


@app.post("/api/vision/start")
async def vision_start(session_id: str, identity_revision: int | None = None):
    """守护开启标记：手动开守护后，帧断流不再被当成'意外断连'。"""
    if not settings.vision_enabled:
        raise HTTPException(503, "坐姿守护未启用")
    person = _vision_person(session_id, identity_revision)
    study.mark_vision_toggle(person['id'], True)
    return {"ok": True}


@app.post("/api/vision/stop")
async def vision_stop(session_id: str, identity_revision: int | None = None):
    """守护关闭标记：孩子主动关守护 ≠ 离座，会话不应被自动暂停。"""
    if not settings.vision_enabled:
        raise HTTPException(503, "坐姿守护未启用")
    person = _vision_person(session_id, identity_revision)
    study.mark_vision_toggle(person['id'], False)
    return {"ok": True}


# ===== 坐姿守护 API（第二期 2A：帧仅内存分析，不落盘不上传） =====

def _vision_person(session_id, identity_revision):
    speaker = check_identity(session_id, identity_revision)
    if not speaker['person']:
        raise HTTPException(400, "需要先选择或识别成员")
    return speaker['person']


async def _vision_image(request_image: UploadFile) -> bytes:
    data = await request_image.read(settings.vision_max_bytes + 1)
    if len(data) > settings.vision_max_bytes:
        raise HTTPException(413, "图片过大，请降低分辨率")
    return data


@app.get("/api/vision/status")
async def vision_status(session_id: str, identity_revision: int | None = None):
    if not settings.vision_enabled:
        return {"enabled": False}
    person = _vision_person(session_id, identity_revision)
    return {"enabled": True, **vision.details(),
            "calibrated": vision_mod.get_baseline(person['id']) is not None,
            "sample_seconds": settings.vision_sample_seconds}


@app.post("/api/vision/calibrate")
async def vision_calibrate(image: UploadFile = File(...), session_id: str = Query(min_length=1, max_length=100),
                           identity_revision: int | None = None):
    """以当前姿态为基线：孩子先坐端正，点校准。"""
    if not settings.vision_enabled:
        raise HTTPException(503, "坐姿守护未启用")
    if not vision.available():
        raise HTTPException(503, "本机未安装姿态模型依赖")
    person = _vision_person(session_id, identity_revision)
    data = await _vision_image(image)
    try:
        metrics, present, quality_ok, person_count = await asyncio.to_thread(vision.analyze_bytes, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        data = None  # 帧立即释放
    if person_count >= 2:
        raise HTTPException(422, "画面里有多个人，请单独正对摄像头校准")
    if not present or not metrics or not quality_ok:
        raise HTTPException(422, "没有看清你（需要正对摄像头、光线充足、头肩完整入镜），请调整后重试")
    gate_error = vision_mod.validate_calibration(metrics)
    if gate_error:
        raise HTTPException(422, gate_error)
    vision_mod.save_baseline(person['id'], metrics)
    vision_mod.judge.reset(person['id'])  # 新基线生效：清零连击/预算/怀疑抑制
    record_event("vision_calibrated", {"person_id": person['id']})
    return {"ok": True, "baseline": metrics}


@app.post("/api/vision/frame")
async def vision_frame(image: UploadFile = File(...), session_id: str = Query(min_length=1, max_length=100),
                       identity_revision: int | None = None):
    """浏览器每 sample_seconds 送一帧：只回数值，只存数值。"""
    if not settings.vision_enabled:
        raise HTTPException(503, "坐姿守护未启用")
    if not vision.available():
        raise HTTPException(503, "本机未安装姿态模型依赖")
    person = _vision_person(session_id, identity_revision)
    sess = study.get_current_session(person['id'])
    if not sess:
        return {"ignored": True, "reason": "没有进行中的学习会话，守护已暂停"}
    baseline = vision_mod.get_baseline(person['id'])
    data = await _vision_image(image)
    try:
        metrics, present, quality_ok, person_count = await asyncio.to_thread(vision.analyze_bytes, data)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        data = None
    multi = person_count >= 2
    severities = vision_mod.issue_severities(metrics, baseline) if (metrics and not multi) else {}
    issues = list(severities)
    action = vision_mod.judge.record(person['id'], sess['id'], severities, present, quality_ok, multi=multi)
    prefix = f"{person['name']}，" if person.get('name') else ''
    speak_texts = [prefix + vision_mod.WARNING_TEXT[i] for i in action['warnings'][:1]]
    if action.get('multi_notice'):
        speak_texts.append("画面里好像有好几个人，我认不准是谁在学习——坐姿提醒先暂停，请单独坐在摄像头前。")
    if action.get('suspect_new'):
        speak_texts.append("我可能看错了你的坐姿，请在页面上重新校准一下。")
        record_event("vision_suspect_calibration", {"person_id": person['id'], "stats": action['stats']})
    if action['left']:
        if sess['status'] == 'active':
            study.pause_session(sess['id'])
        speak_texts.append("检测到你不在了，计时先暂停了。")
    if action['returned']:
        current = study.get_session(sess['id'])
        if current and current['status'] == 'paused':
            study.resume_session(sess['id'])
        speak_texts.append("欢迎回来，计时继续。")
    warning_text = " ".join(speak_texts) if speak_texts else None
    # 之前提醒过的问题本轮消失：3 分钟窗口内记为已响应（多人帧归属不明，不参与）
    if present and metrics and not multi:
        vision_mod.mark_responded(person['id'], [k for k in vision_mod.WARNING_TEXT if k not in issues])
    vision_mod.log_frame(sess['id'], person['id'], present, metrics, issues,
                         action['warnings'], warning_text, quality='multi' if multi else None)
    if speak_texts:
        # 姿态警告走统一通道（P3）；到离座/多人/校准类状态播报优先级更高（P2）
        prio = (speech.PRIORITY_STUDY
                if (action['left'] or action['returned'] or action.get('multi_notice') or action.get('suspect_new'))
                else speech.PRIORITY_POSTURE)
        res = speech_channel.submit(warning_text, prio, fresh_seconds=60)
        record_event("vision_warning", {"person_id": person['id'], "text": warning_text, "speech": res})
    return {"present": present, "quality_ok": quality_ok, "multi": multi,
            "person_count": person_count, "issues": issues,
            "severe": [i for i, s in severities.items() if s == 'severe'],
            "warnings": speak_texts, "paused": action['left'], "resumed": action['returned'],
            "suppressed": action.get('suppressed'), "stats": action.get('stats'),
            "calibrated": baseline is not None}


# ===== 统一语音通道 API（2026-11-13 第一批） =====

class DndIn(BaseModel):
    minutes: int | None = Field(default=None, ge=1, le=480)


@app.get("/api/speech/status")
async def speech_status():
    """勿扰/安静时段/静音/队列/最近被压制的话——前端横幅数据源。"""
    return speech_channel.status()


@app.post("/api/speech/dnd")
async def speech_dnd_on(payload: DndIn | None = None):
    minutes = (payload.minutes if payload and payload.minutes else settings.dnd_default_minutes)
    until = speech.set_dnd(minutes)
    record_event("dnd_set", {"minutes": minutes, "until": until})
    return speech_channel.status()


@app.delete("/api/speech/dnd")
async def speech_dnd_off():
    speech.clear_dnd()
    record_event("dnd_cleared", {})
    return speech_channel.status()
