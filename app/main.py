from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .db import init_db, list_events, record_event
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
        try:
            for reminder in due_now():
                person = f"{reminder['person_name']}，" if reminder.get("person_name") else ""
                spoken = f"{person}提醒时间到了：{reminder['text']}"
                result = voice.speak(spoken)
                record_event("reminder_delivered", {"reminder_id": reminder["id"], "text": spoken, "voice": result})
        except Exception as exc:
            # A broken audio driver must not stop the reminder scheduler.
            record_event("reminder_error", {"error": str(exc)})
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
    return """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>家庭机器人</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172033;background:#f5f7fb}*{box-sizing:border-box}body{max-width:1100px;margin:0 auto;padding:28px 18px}h1{margin:0;font-size:30px}.sub{color:#697386;margin:8px 0 24px}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}.card{background:white;border:1px solid #e6eaf1;border-radius:18px;padding:20px;box-shadow:0 5px 18px #1b2d4b0a}h2{font-size:18px;margin:0 0 14px}.chat{min-height:280px;display:flex;flex-direction:column}.messages{flex:1;min-height:160px;overflow:auto;background:#f8fafc;border-radius:12px;padding:12px}.msg{padding:9px 12px;border-radius:12px;margin:7px 0;max-width:85%;white-space:pre-wrap}.user{background:#dcecff;margin-left:auto}.bot{background:#eaf7ed}.row{display:flex;gap:8px;margin-top:10px}input,button,textarea{font:inherit;border-radius:10px;border:1px solid #d5dbe6;padding:10px}input,textarea{width:100%}button{background:#2764e8;color:white;border:0;cursor:pointer}button.secondary{background:#eef2f8;color:#26344d}.small{font-size:13px;color:#697386}.reminder{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #edf0f5}.camera{width:100%;border-radius:12px;background:#131a27;aspect-ratio:4/3;object-fit:cover}.status{padding:8px 10px;border-radius:9px;background:#f0f4fa;color:#4e5c70;margin-top:10px}.ok{color:#16834b}.warn{color:#a45b00}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body><h1>家庭机器人</h1><div class="sub">先让它在电脑上听懂、看见、记住和提醒，再迁移到树莓派。</div>
<div class="grid"><section class="card chat"><h2>和机器人说话</h2><div id="messages" class="messages"><div class="msg bot">你好，我可以和你聊天，也可以帮你设置提醒。</div></div><div class="row"><input id="chatInput" placeholder="例如：提醒我明天晚上八点练琴" autocomplete="off"><button id="micBtn" class="secondary" title="使用浏览器麦克风">🎙️</button><button id="sendBtn">发送</button></div><div id="voiceStatus" class="small">支持 Chrome/Safari 的浏览器语音识别；也可以直接输入文字。</div></section>
<section class="card"><h2>提醒</h2><form id="reminderForm"><div class="row"><input id="reminderText" placeholder="提醒内容" required><input id="due" type="datetime-local" required></div><button style="margin-top:10px">创建提醒</button></form><div id="reminders" class="small" style="margin-top:14px">加载中…</div></section>
<section class="card"><h2>摄像头与家庭成员</h2><video id="camera" class="camera" autoplay muted playsinline></video><canvas id="snapshot" hidden></canvas><div class="row"><button id="startCamera" class="secondary">打开摄像头</button><button id="recognize">识别我</button></div><div class="row"><input id="personName" placeholder="姓名（注册时填写）"><button id="enroll" class="secondary">拍照注册</button></div><div id="visionStatus" class="status">摄像头尚未打开</div></section>
<section class="card"><h2>最近发生的事</h2><div id="events" class="small">加载中…</div><div id="systemStatus" class="status">检查服务中…</div></section></div>
<script>
const $=id=>document.getElementById(id); const messages=$('messages');
function addMsg(text,who='bot'){const d=document.createElement('div');d.className='msg '+who;d.textContent=text;messages.appendChild(d);messages.scrollTop=messages.scrollHeight}
function say(text){if('speechSynthesis' in window){speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(text);u.lang='zh-CN';speechSynthesis.speak(u)}}
async function send(){const text=$('chatInput').value.trim();if(!text)return;addMsg(text,'user');$('chatInput').value='';try{const r=await fetch('/api/command?text='+encodeURIComponent(text),{method:'POST'});const data=await r.json();if(!r.ok)throw Error(data.detail||'请求失败');if(data.needs_confirmation){addMsg(data.message);say(data.message);const b=document.createElement('button');b.textContent='确认创建提醒';b.onclick=async()=>{const x=await fetch('/api/reminders',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:data.text,due_at:data.due_at})});if(x.ok){addMsg('好的，提醒已经记下了。');say('好的，提醒已经记下了。');b.remove();loadReminders()}};messages.appendChild(b)}else{addMsg(data.reply||'我暂时没有听懂。');say(data.reply||'我暂时没有听懂。')} }catch(e){addMsg('暂时出错：'+e.message)}}
$('sendBtn').onclick=send;$('chatInput').onkeydown=e=>{if(e.key==='Enter')send()};
async function loadReminders(){const r=await fetch('/api/reminders');const a=await r.json();$('reminders').innerHTML=a.length?a.map(x=>`<div class="reminder"><span>${new Date(x.due_at).toLocaleString()}<br>${x.text}</span><button class="secondary" onclick="cancelReminder(${x.id})">取消</button></div>`).join(''):'暂无待提醒事项'}
async function cancelReminder(id){await fetch('/api/reminders/'+id,{method:'DELETE'});loadReminders()}
$('reminderForm').onsubmit=async e=>{e.preventDefault();const r=await fetch('/api/reminders',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('reminderText').value,due_at:new Date($('due').value).toISOString()})});if(r.ok){$('reminderText').value='';loadReminders();addMsg('提醒已经创建。');}}
let stream;async function openCamera(){try{stream=await navigator.mediaDevices.getUserMedia({video:true});$('camera').srcObject=stream;$('visionStatus').textContent='摄像头已打开，可以识别或注册';$('visionStatus').className='status ok'}catch(e){$('visionStatus').textContent='无法打开摄像头：'+e.message;$('visionStatus').className='status warn'}}$('startCamera').onclick=openCamera;
async function photo(endpoint,extra={}){if(!stream)return alert('请先打开摄像头');const v=$('camera'),c=$('snapshot');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);const blob=await new Promise(ok=>c.toBlob(ok,'image/jpeg',.9));const f=new FormData();f.append('image',blob,'camera.jpg');Object.entries(extra).forEach(([k,v])=>f.append(k,v));const r=await fetch(endpoint,{method:'POST',body:f});const data=await r.json();if(!r.ok)throw Error(data.detail||'视觉处理失败');return data}
$('recognize').onclick=async()=>{try{const d=await photo('/api/faces/recognize');const names=d.faces.map(x=>x.name).join('、')||'没有检测到人脸';$('visionStatus').textContent='识别结果：'+names;$('visionStatus').className='status ok';say(names==='没有检测到人脸'?names:'你好，'+names)}catch(e){$('visionStatus').textContent=e.message;$('visionStatus').className='status warn'}};
$('enroll').onclick=async()=>{const name=$('personName').value.trim();if(!name)return alert('请先填写姓名');try{const d=await photo('/api/faces/enroll',{name});$('visionStatus').textContent='已注册：'+d.name;$('visionStatus').className='status ok'}catch(e){$('visionStatus').textContent=e.message;$('visionStatus').className='status warn'}};
async function loadEvents(){const a=await (await fetch('/api/events')).json();$('events').innerHTML=a.length?a.slice(0,8).map(x=>`<div class="reminder"><span>${new Date(x.created_at).toLocaleTimeString()} ${x.payload.text||x.type}</span></div>`).join(''):'暂无事件'}
async function health(){const d=await (await fetch('/api/health')).json();$('systemStatus').innerHTML='<span class="ok">● 服务正常</span>　模型：'+(d.llm_enabled?'已连接':'未配置')}
async function refresh(){loadReminders();loadEvents();health()}refresh();setInterval(refresh,10000);
let recognition; if('webkitSpeechRecognition' in window||'SpeechRecognition' in window){const R=window.SpeechRecognition||window.webkitSpeechRecognition;recognition=new R();recognition.lang='zh-CN';recognition.interimResults=false;recognition.onstart=()=>{$('voiceStatus').textContent='正在听…';};recognition.onresult=e=>{$('chatInput').value=e.results[0][0].transcript;send()};recognition.onerror=e=>{$('voiceStatus').textContent='语音识别失败：'+e.error};recognition.onend=()=>{$('voiceStatus').textContent='可以继续说话或输入文字'};$('micBtn').onclick=()=>recognition.start()}else{$('micBtn').disabled=true;$('voiceStatus').textContent='当前浏览器不支持语音识别，请使用文字输入'}
</script></body></html>"""


@app.get("/api/health")
def health():
    return {"ok": True, "service": "family-robot", "version": app.version, "llm_enabled": llm.enabled, "llm_blocked": llm.blocked_reason, "voice": voice.status()}


@app.get("/api/reminders")
def reminders():
    return list_active()


@app.get("/api/events")
def events():
    return list_events()


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
def chat(text: str):
    if not llm.enabled:
        raise HTTPException(503, "LLM 未配置")
    try:
        return {"reply": llm.chat(text), "model": settings.llm_model}
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"模型调用失败：{exc}") from exc
