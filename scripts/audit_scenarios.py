"""阿伴家庭机器人 · 跨功能连接场景审计（真实运行实例，非 mock）。

把"前端会走的真实接口序列"连起来跑，断言用户可感知的结果，记录证据。
覆盖：访客聊天+语音 / 自然语言提醒全链路 / 成员记忆与归属 / 学习陪伴 /
勿扰闭环 / 音色切换 / 错误契约 / 身份版本冲突 / 新对话重置 / 删除保护 /
重复提醒 NL 说法电池 / 前端 DOM 接线。

⚠️ 必须对隔离实例运行，会创建测试成员/提醒/草稿，切勿打生产库：
    mkdir -p /tmp/robot-audit-data
    cd /Users/yuanlw/robot
    ROBOT_DATA_DIR=/tmp/robot-audit-data ROBOT_PORT=8010 ./run.sh &   # 另起终端
    .venv-voice/bin/python scripts/audit_scenarios.py \
        --base-url http://127.0.0.1:8010 --data-dir /tmp/robot-audit-data
默认拒绝打 :8000（生产默认端口），除非显式 --force。

送达类检查自适应系统静音状态：
  - 未静音：断言到点 delivered（提醒不漏）；
  - 静音中：断言压制契约（reminder_suppressed 且未 delivered、提醒保持 active）。
两种都是正确行为，分别验证。
"""
import argparse
import array
import io
import json
import math
import re
import sqlite3
import sys
import time
import uuid
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

TZ = timezone(timedelta(hours=8))
ROOT = Path(__file__).resolve().parent.parent
RESULTS = []


def rec(jid, name, ok, detail=""):
    RESULTS.append((jid, name, bool(ok), detail))
    print(("PASS" if ok else "FAIL") + f" [{jid}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def now():
    return datetime.now(TZ)


def iso(dt):
    return dt.isoformat(timespec="seconds")


def wav_ok(data):
    if not data.startswith(b"RIFF"):
        return False, "not WAV"
    with wave.open(io.BytesIO(data)) as w:
        sec = w.getnframes() / w.getframerate()
        s = array.array("h"); s.frombytes(w.readframes(w.getnframes()))
    rms = math.sqrt(sum(x * x for x in s) / max(len(s), 1))
    return (sec > 0.3 and rms > 100), f"{sec:.1f}s RMS={rms:.0f}"


class Client:
    def __init__(self, base):
        self.c = httpx.Client(base_url=base, timeout=120, headers={"X-Robot-Client": "audit"})

    def get(self, p, **kw): return self.c.get(p, **kw)
    def post(self, p, **kw): return self.c.post(p, **kw)
    def put(self, p, **kw): return self.c.put(p, **kw)
    def patch(self, p, **kw): return self.c.patch(p, **kw)
    def delete(self, p, **kw): return self.c.delete(p, **kw)

    def stream(self, text, sid, rev=None):
        params = {"text": text, "session_id": sid}
        if rev is not None:
            params["identity_revision"] = rev
        events = []
        with self.c.stream("POST", "/api/command/stream", params=params) as r:
            if r.status_code != 200:
                return events, r.status_code
            for line in r.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events, 200

    def reply_of(self, events):
        return "".join(e.get("text", "") for e in events if e.get("type") == "delta").strip()

    def identity(self, sid): return self.get(f"/api/conversations/{sid}/identity").json()
    def select(self, sid, pid, rev): return self.put(f"/api/conversations/{sid}/identity", json={"person_id": pid, "revision": rev})
    def events(self): return self.get("/api/events").json()
    def muted(self): return self.get("/api/speech/status").json().get("muted") is True


def wait_event(cli, etype, match=None, timeout=100, poll=3):
    end = time.time() + timeout
    while time.time() < end:
        for e in cli.events():
            if e["type"] == etype and (match is None or match(e.get("payload", {}))):
                return e
        time.sleep(poll)
    return None


def delivery_check(cli, jid, name, rid, text_match, timeout=110):
    """自适应静音：未静音验 delivered；静音验压制契约（不 delivered、保持 active）。"""
    if cli.muted():
        sup = wait_event(cli, "reminder_suppressed", lambda p: p.get("reminder_id") == rid, timeout=timeout)
        delivered = any(e["type"] == "reminder_delivered" and e["payload"].get("reminder_id") == rid for e in cli.events())
        active = any(r["id"] == rid and r["status"] == "active" for r in cli.get("/api/reminders").json())
        rec(jid, name + "（静音→压制契约）", sup is not None and not delivered and active,
            f"suppressed={sup is not None} delivered={delivered} active={active}")
    else:
        evd = wait_event(cli, "reminder_delivered", lambda p: text_match(p.get("text", "")), timeout=timeout)
        rec(jid, name, evd is not None, f"event={evd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8010")
    ap.add_argument("--data-dir", default="/tmp/robot-audit-data")
    ap.add_argument("--skip-wait", action="store_true", help="跳过到点送达等待")
    ap.add_argument("--force", action="store_true", help="允许打生产端口（危险，会污染数据）")
    args = ap.parse_args()

    if args.base_url.rstrip("/").endswith(":8000") and not args.force:
        print("拒绝：:8000 是生产默认端口。请对隔离实例运行（见脚本头部用法），或显式 --force。")
        return 2
    cli = Client(args.base_url)

    db = sqlite3.connect(args.data_dir + "/robot.db", timeout=15)
    tag = uuid.uuid4().hex[:4]
    jia_name, yi_name = f"审计甲{tag}", f"审计乙{tag}"
    db.execute("INSERT OR IGNORE INTO people(name,face_encoding,created_at) VALUES(?,NULL,?)", (jia_name, iso(now())))
    db.execute("INSERT OR IGNORE INTO people(name,face_encoding,created_at) VALUES(?,NULL,?)", (yi_name, iso(now())))
    db.commit(); db.close()
    people = {p["name"]: p["id"] for p in cli.get("/api/people").json()}
    jia, yi = people.get(jia_name), people.get(yi_name)
    rec("SETUP", "测试成员就绪", bool(jia and yi), f"甲={jia} 乙={yi} muted={cli.muted()}")

    # J1 访客聊天 + 语音
    try:
        s1 = "audit-" + uuid.uuid4().hex[:8]
        ident = cli.identity(s1)
        rec("J1a", "访客身份初始为空", ident["person"] is None, str(ident))
        t0 = time.time()
        ev, code = cli.stream("你好，请用一句话介绍你自己", s1, ident["revision"])
        reply = cli.reply_of(ev); types = [e["type"] for e in ev]
        rec("J1b", "聊天流式有回答且正常收尾", code == 200 and reply and "done" in types and "error" not in types,
            f"{time.time()-t0:.1f}s reply={reply[:50]!r}")
        r = cli.post("/api/voice/audio", json={"text": reply[:120] or "你好"})
        ok, stat = wav_ok(r.content)
        rec("J1c", "回答能合成非静音语音", r.status_code == 200 and ok, f"HTTP{r.status_code} {stat}")
    except Exception as e:
        rec("J1", "访客聊天+语音", False, repr(e))

    # J2 自然语言提醒 → 草稿 → 确认 → 送达
    try:
        s2 = "audit-" + uuid.uuid4().hex[:8]
        rev = cli.identity(s2)["revision"]
        ev, code = cli.stream("一分钟后提醒我喝水", s2, rev)
        result = next((e for e in ev if e.get("type") == "result" and e.get("needs_confirmation")), None)
        rec("J2a", "自然语言识别为提醒草稿", code == 200 and result is not None, f"result={bool(result)}")
        if result:
            drafts = cli.get(f"/api/conversations/{s2}/drafts").json()
            d = next((x for x in drafts if "喝水" in x.get("text", "")), drafts[0] if drafts else None)
            rec("J2b", "草稿可查询", d is not None, f"drafts={len(drafts)}")
            if d:
                cr = cli.post(f"/api/reminder-drafts/{d['id']}/confirm", json={"session_id": s2, "revision": d["revision"], "identity_revision": rev})
                rec("J2c", "确认草稿落库", cr.status_code == 200, f"HTTP{cr.status_code}")
                rem = cli.get("/api/reminders", params={"session_id": s2}).json()
                rid = next((x["id"] for x in rem if "喝水" in x["text"]), None)
                rec("J2d", "提醒出现在列表", rid is not None, f"rid={rid}")
                if not args.skip_wait and rid:
                    delivery_check(cli, "J2e", "到点真实送达（提醒不漏）", rid, lambda t: "喝水" in t)
    except Exception as e:
        rec("J2", "自然语言提醒全链路", False, repr(e))

    # J3 成员 + 记忆跨浏览器 + 归属
    try:
        sa = "audit-" + uuid.uuid4().hex[:8]
        sel = cli.select(sa, jia, cli.identity(sa)["revision"])
        rec("J3a", "选择成员 revision 递增", sel.status_code == 200 and sel.json()["person"]["id"] == jia, sel.text[:80])
        rev_a = sel.json()["revision"]
        ev, _ = cli.stream("请记住我最喜欢喝普洱茶", sa, rev_a)
        rec("J3b", "写入个人记忆", cli.reply_of(ev) != "", cli.reply_of(ev)[:40])
        sb = "audit-" + uuid.uuid4().hex[:8]
        selb = cli.select(sb, jia, cli.identity(sb)["revision"])
        ev2, _ = cli.stream("我最喜欢喝什么茶？", sb, selb.json()["revision"])
        rec("J3c", "换浏览器仍记得个人记忆", "普洱" in cli.reply_of(ev2), cli.reply_of(ev2)[:50])
        due = iso(now() + timedelta(hours=2))
        pr = cli.post("/api/reminders", params={"session_id": sa, "identity_revision": rev_a}, json={"text": "甲的私人提醒", "due_at": due, "scope": "personal"})
        pid_rem = pr.json().get("id")
        rec("J3d", "创建个人提醒", pr.status_code == 200, f"HTTP{pr.status_code}")
        sc = "audit-" + uuid.uuid4().hex[:8]
        rev_c = cli.select(sc, yi, cli.identity(sc)["revision"]).json()["revision"]
        visible = cli.get("/api/reminders", params={"session_id": sc, "identity_revision": rev_c}).json()
        rec("J3e", "他人看不到甲的私人提醒", not any(x["id"] == pid_rem for x in visible), "")
        if pid_rem:
            rec("J3f", "他人不能删甲的私人提醒", cli.delete(f"/api/reminders/{pid_rem}", params={"session_id": sc, "identity_revision": rev_c}).status_code == 404, "")
        hr = cli.post("/api/reminders", params={"session_id": sa, "identity_revision": rev_a}, json={"text": "全家共享提醒", "due_at": due, "scope": "household"})
        vis2 = cli.get("/api/reminders", params={"session_id": sc, "identity_revision": rev_c}).json()
        rec("J3g", "家庭共享提醒全员可见", hr.status_code == 200 and any("全家共享" in x["text"] for x in vis2), "")
    except Exception as e:
        rec("J3", "成员/记忆/归属", False, repr(e))

    # J4 学习陪伴
    try:
        ss = "audit-" + uuid.uuid4().hex[:8]
        rev_s = cli.select(ss, jia, cli.identity(ss)["revision"]).json()["revision"]
        st = cli.post("/api/study/start", params={"session_id": ss, "identity_revision": rev_s})
        rec("J4a", "开始学习会话", st.status_code == 200, f"HTTP{st.status_code}")
        cur = cli.get("/api/study/current", params={"session_id": ss, "identity_revision": rev_s}).json()
        rec("J4b", "查询当前学习状态", cur.get("session") is not None, "")
        rate = cli.post("/api/study/rate", json={"session_id": ss, "score": 8, "tags": ["晚上"], "identity_revision": rev_s})
        rec("J4c", "学习中可自评", rate.status_code == 200, f"HTTP{rate.status_code}")
        en = cli.post("/api/study/end", params={"session_id": ss, "identity_revision": rev_s})
        rec("J4d", "结束学习有总结", en.status_code == 200 and "summary" in en.json(), "")
        rep = cli.get("/api/study/report", params={"session_id": ss, "period": "week", "identity_revision": rev_s})
        rec("J4e", "学习报告有内容", rep.status_code == 200 and rep.json().get("markdown"), "")
    except Exception as e:
        rec("J4", "学习陪伴", False, repr(e))

    # J5 勿扰闭环
    try:
        sd = "audit-" + uuid.uuid4().hex[:8]
        rev_d = cli.identity(sd)["revision"]
        on = cli.post("/api/speech/dnd", json={"minutes": 5})
        rec("J5a", "开启勿扰", on.status_code == 200 and on.json()["dnd_active"], "")
        rr = cli.post("/api/reminders", params={"session_id": sd, "identity_revision": rev_d}, json={"text": "勿扰测试提醒", "due_at": iso(now() + timedelta(seconds=18)), "scope": "household"})
        rid = rr.json().get("id")
        if not args.skip_wait and rid:
            sup = wait_event(cli, "reminder_suppressed", lambda p: p.get("reminder_id") == rid and p.get("reason") == "deferred_dnd", timeout=55)
            rec("J5b", "勿扰期间提醒被压制", sup is not None, f"event={sup}")
            off = cli.delete("/api/speech/dnd")
            rec("J5c", "解除勿扰", off.status_code == 200 and not off.json()["dnd_active"], "")
            delivery_check(cli, "J5d", "解除后提醒补达（不漏）", rid, lambda t: "勿扰测试" in t, timeout=60)
        else:
            cli.delete("/api/speech/dnd")
    except Exception as e:
        rec("J5", "勿扰闭环", False, repr(e))

    # J6 音色切换
    try:
        vs = cli.get("/api/voice/settings").json()
        rec("J6a", "音色列表可用", len(vs.get("available_voices", [])) >= 2, "")
        if any(v["id"] == "argmax:serena" for v in vs.get("available_voices", [])):
            sw = cli.put("/api/voice/settings", json={"voice": "argmax:serena"})
            au = cli.post("/api/voice/audio", json={"text": "音色切换测试。"})
            ok, stat = wav_ok(au.content)
            rec("J6b", "Argmax 合成非静音", sw.status_code == 200 and au.status_code == 200 and ok, stat)
        rec("J6c", "非法音色被拒", cli.put("/api/voice/settings", json={"voice": "不存在"}).status_code == 422, "")
        cli.put("/api/voice/settings", json={"voice": "bailian:Cherry"})
    except Exception as e:
        rec("J6", "音色切换", False, repr(e))

    # J7 错误契约
    try:
        no_csrf = httpx.post(args.base_url + "/api/reminders", json={"text": "x", "due_at": iso(now() + timedelta(hours=1))}, timeout=30)
        rec("J7a", "缺 CSRF 头被拒", no_csrf.status_code == 403, f"HTTP{no_csrf.status_code}")
        rec("J7b", "空指令被拒", cli.post("/api/command/stream", params={"text": "", "session_id": "x"}).status_code == 422, "")
        rec("J7c", "过去时间提醒被拒", cli.post("/api/reminders", json={"text": "过去", "due_at": iso(now() - timedelta(hours=1))}).status_code == 422, "")
        rec("J7d", "无时区提醒被拒", cli.post("/api/reminders", json={"text": "无时区", "due_at": "2026-12-01T08:00:00"}).status_code == 422, "")
        sr = 16000; buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(b"\x00\x00" * sr)
        stt = cli.post("/api/stt", files={"audio": ("s.wav", buf.getvalue(), "audio/wav")})
        rec("J7e", "全静音录音门控", stt.status_code == 200 and stt.json().get("silent") is True, "")
        rec("J7f", "非法录音被拒", cli.post("/api/stt", files={"audio": ("s.wav", b"not-wav", "audio/wav")}).status_code == 422, "")
        rec("J7g", "未选身份不能开始学习", cli.post("/api/study/start", params={"session_id": "audit-nobody-" + uuid.uuid4().hex[:6]}).status_code == 400, "")
    except Exception as e:
        rec("J7", "错误契约", False, repr(e))

    # J8 身份版本冲突 + 新对话重置 + 删除保护
    try:
        s9 = "audit-" + uuid.uuid4().hex[:8]
        r0 = cli.identity(s9)["revision"]
        cli.select(s9, jia, r0)
        rec("J8a", "过期 revision 切换被拒", cli.select(s9, yi, r0).status_code == 409, "")
        s10 = "audit-" + uuid.uuid4().hex[:8]
        rev = cli.select(s10, jia, cli.identity(s10)["revision"]).json()["revision"]
        cli.stream("请记住我的幸运数字是42", s10, rev)
        rec("J8b", "新对话重置成功", cli.delete(f"/api/conversations/{s10}", params={"identity_revision": rev}).status_code == 200, "")
        ev, _ = cli.stream("我的幸运数字是多少？", s10, rev)
        rec("J8c", "重置后不再记得旧记忆", "42" not in cli.reply_of(ev), cli.reply_of(ev)[:50])
        s11 = "audit-" + uuid.uuid4().hex[:8]
        rev11 = cli.select(s11, yi, cli.identity(s11)["revision"]).json()["revision"]
        rr = cli.post("/api/reminders", params={"session_id": s11, "identity_revision": rev11}, json={"text": "乙未完成提醒", "due_at": iso(now() + timedelta(hours=3)), "scope": "personal"})
        rid11 = rr.json().get("id")
        rec("J8d", "有未完成个人提醒拒删成员", cli.delete(f"/api/people/{yi}").status_code == 409, "")
        if rid11:
            cli.delete(f"/api/reminders/{rid11}", params={"session_id": s11, "identity_revision": rev11})
        rec("J8e", "清空提醒后可删成员", cli.delete(f"/api/people/{yi}").status_code == 200, "")
        rec("J8f", "被删成员浏览器退回访客", cli.identity(s11)["person"] is None, "")
    except Exception as e:
        rec("J8", "身份/重置/删除", False, repr(e))

    # J9 重复提醒 NL 说法电池（真实模型）
    try:
        for text, want in [("每天晚上八点提醒我练琴", "daily"), ("每周一早上九点提醒我开会", "weekly"), ("提醒我明天晚上八点练琴", "once")]:
            sid = "nl-" + uuid.uuid4().hex[:8]
            rev = cli.identity(sid)["revision"]
            cli.stream(text, sid, rev)
            drafts = cli.get(f"/api/conversations/{sid}/drafts").json()
            got = drafts[0].get("repeat_rule") if drafts else None
            rec("J9", f"NL重复提醒[{text[:12]}]", got == want, f"repeat={got} want={want}")
        # 歧义不误承诺：无时间应追问而非建草稿
        sid = "nl-" + uuid.uuid4().hex[:8]
        cli.stream("提醒我吃药", sid, cli.identity(sid)["revision"])
        rec("J9x", "无时间提醒追问不建草稿", len(cli.get(f"/api/conversations/{sid}/drafts").json()) == 0, "")
    except Exception as e:
        rec("J9", "NL重复提醒电池", False, repr(e))

    # J10 前端 DOM 接线一致性
    try:
        html = (ROOT / "app" / "web.html").read_text(encoding="utf-8")
        declared = set(re.findall(r'id="([^"]+)"', html))
        used = set(re.findall(r"\$\('([^']+)'\)", html))
        missing = sorted(used - declared)
        rec("J10a", "JS 引用的元素 id 全部存在", not missing, f"missing={missing}")
        panes = set(re.findall(r'data-pane="([^"]+)"', html))
        pane_ids = set(re.findall(r'<main id="([^"]+)" class="pane', html))
        rec("J10b", "导航目标全部存在", not (panes - pane_ids), f"missing={sorted(panes-pane_ids)}")
    except Exception as e:
        rec("J10", "DOM 接线", False, repr(e))

    print("\n========== 审计汇总 ==========", flush=True)
    passed = sum(1 for r in RESULTS if r[2]); total = len(RESULTS)
    for jid, name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAIL [{jid}] {name}: {detail}", flush=True)
    print(f"通过 {passed}/{total}", flush=True)
    print("AUDIT_RESULT=" + ("PASS" if passed == total else f"FAIL({total-passed})"), flush=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
