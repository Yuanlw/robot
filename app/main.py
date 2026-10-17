from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .db import init_db
from .commands import parse_reminder
from .face import FaceService
from .reminders import cancel, create, due_now, list_active
from .voice import VoiceService
from .llm import LLMService
from .config import settings

faces = FaceService()
voice = VoiceService()
llm = LLMService()


async def reminder_loop():
    while True:
        for reminder in due_now():
            person = f"{reminder['person_name']}，" if reminder.get("person_name") else ""
            voice.speak(f"{person}提醒时间到了：{reminder['text']}")
        await asyncio.sleep(15)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    task = asyncio.create_task(reminder_loop())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


app = FastAPI(title="家庭机器人", version="0.1.0", lifespan=lifespan)


class ReminderIn(BaseModel):
    text: str
    due_at: datetime
    person_id: int | None = None
    repeat_rule: str = "once"


@app.get("/", response_class=HTMLResponse)
def home():
    return """<!doctype html><meta charset='utf-8'><title>家庭机器人</title>
    <h1>家庭机器人 v0.1</h1><p>先从提醒开始，再接入语音和视觉。</p>
    <h2>提醒</h2><form id='f'><input id='text' placeholder='提醒内容' required><input id='due' type='datetime-local' required><button>创建提醒</button></form><pre id='out'></pre>
    <script>async function load(){out.textContent=JSON.stringify(await (await fetch('/api/reminders')).json(),null,2)}f.onsubmit=async e=>{e.preventDefault();await fetch('/api/reminders',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text.value,due_at:new Date(due.value).toISOString()})});load()};load()</script>"""


@app.get("/api/health")
def health():
    return {"ok": True, "service": "family-robot", "version": app.version, "llm_enabled": llm.enabled}


@app.get("/api/reminders")
def reminders():
    return list_active()


@app.post("/api/reminders")
def add_reminder(payload: ReminderIn):
    return create(payload.text, payload.due_at, payload.person_id, payload.repeat_rule)


@app.delete("/api/reminders/{reminder_id}")
def delete_reminder(reminder_id: int):
    if not cancel(reminder_id):
        raise HTTPException(404, "提醒不存在或已取消")
    return {"ok": True}


@app.post("/api/faces/enroll")
async def enroll_face(name: str = Form(...), image: UploadFile = File(...)):
    path = Path("/tmp") / f"robot-enroll-{image.filename}"
    path.write_bytes(await image.read())
    try:
        return faces.enroll(name, str(path))
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/faces/recognize")
async def recognize_face(image: UploadFile = File(...)):
    path = Path("/tmp") / f"robot-recognize-{image.filename}"
    path.write_bytes(await image.read())
    try:
        return faces.recognize(str(path))
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/speak")
def speak(text: str):
    return {"text": text, "status": voice.speak(text)}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    path = Path("/tmp") / f"robot-audio-{audio.filename}"
    path.write_bytes(await audio.read())
    try:
        return {"text": voice.transcribe(str(path))}
    except RuntimeError as exc:
        raise HTTPException(501, str(exc)) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/api/command")
def command(text: str):
    parsed = parse_reminder(text)
    if parsed:
        reminder_text, due_at = parsed
        return {"needs_confirmation": True, "text": reminder_text, "due_at": due_at.isoformat(), "message": f"确认一下，{due_at:%m月%d日%H点%M分}提醒你{reminder_text}，可以吗？", "source": "local"}
    if llm.enabled:
        try:
            return {"needs_confirmation": False, "reply": llm.chat(text), "source": "llm"}
        except Exception as exc:
            raise HTTPException(502, f"模型调用失败：{exc}") from exc
    raise HTTPException(400, "目前支持的格式：提醒我明天晚上八点练琴；复杂表达需要配置模型")


@app.post("/api/chat")
def chat(text: str):
    if not llm.enabled:
        raise HTTPException(503, "LLM 未配置")
    try:
        return {"reply": llm.chat(text), "model": settings.llm_model}
    except Exception as exc:
        raise HTTPException(502, f"模型调用失败：{exc}") from exc
