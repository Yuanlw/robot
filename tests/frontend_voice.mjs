// 学习模式纯语音对话前端逻辑测试（node 真跑 web.html 的语音函数，不依赖真实麦克风/WS）。
// 用法：node tests/frontend_voice.mjs （由 tests/test_frontend_voice.py 接入 unittest）
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const html = fs.readFileSync(path.join(root, 'app', 'web.html'), 'utf8');
function extract(name) {
  const start = html.indexOf('function ' + name);
  if (start < 0) throw new Error('找不到函数 ' + name);
  return html.slice(start, html.indexOf('\n}', start) + 2);
}
const src = ['handleVoiceMsg', 'playVoiceAudio', 'stopVoiceMode'].map(extract).join('\n');

let failed = 0;
function check(name, cond, detail) {
  console.log((cond ? 'PASS: ' : 'FAIL: ') + name + (cond ? '' : ' —— ' + detail));
  if (!cond) failed++;
}

function makeEnv() {
  const calls = { state: [], msgs: [], sent: [], btnText: [], status: [] };
  globalThis.$ = (id) => ({
    set textContent(v) { if (id === 'voiceModeBtn') calls.btnText.push(v); if (id === 'voiceModeStatus') calls.status.push(v); },
    get textContent() { return ''; },
    set disabled(v) {}, get disabled() { return false; },
  });
  globalThis.setRobotState = (s) => calls.state.push(s);
  globalThis.addMsg = (t, who) => calls.msgs.push([t, who]);
  globalThis.confirmation = () => {};
  globalThis.lastReply = '';
  globalThis.robotState = 'listening';
  globalThis.micMode = 'worklet';
  globalThis.voiceMode = true;
  globalThis.voiceWs = { readyState: 1, send: (m) => calls.sent.push(m), close: () => {} };
  globalThis.voiceAudio = null;
  globalThis.URL.createObjectURL = () => 'blob:x';
  globalThis.URL.revokeObjectURL = () => {};
  globalThis.Audio = class {
    constructor() { this.onended = null; }
    play() { queueMicrotask(() => this.onended && this.onended()); return Promise.resolve(); }
    pause() {}
  };
  const fns = eval(src + '\n({handleVoiceMsg, playVoiceAudio, stopVoiceMode})');
  return { calls, ...fns };
}

// 1) 状态消息映射
{
  const env = makeEnv();
  env.handleVoiceMsg({ type: 'state', state: 'thinking' });
  env.handleVoiceMsg({ type: 'state', state: 'speaking' });
  check('thinking/speaking 状态映射', env.calls.state.includes('thinking') && env.calls.state.includes('speaking'), JSON.stringify(env.calls.state));
}
// 2) 转录与回答进气泡
{
  const env = makeEnv();
  env.handleVoiceMsg({ type: 'transcript', text: '你好' });
  env.handleVoiceMsg({ type: 'reply', text: '你好呀', drafts: [] });
  check('转录显示为用户气泡', env.calls.msgs.some(m => m[0] === '你好' && m[1] === 'user'), JSON.stringify(env.calls.msgs));
  check('回答显示为机器人气泡', env.calls.msgs.some(m => m[0] === '你好呀' && m[1] === 'bot'), JSON.stringify(env.calls.msgs));
}
// 3) 播放完毕回 playback_done
{
  const env = makeEnv();
  env.playVoiceAudio({});
  await new Promise(r => setImmediate(r));
  check('播放完毕发送 playback_done', env.calls.sent.some(m => JSON.parse(m).type === 'playback_done'), JSON.stringify(env.calls.sent));
}
// 4) 退出复位
{
  const env = makeEnv();
  env.stopVoiceMode();
  check('退出后按钮文案复位', env.calls.btnText.some(t => t.includes('开启纯语音对话')), JSON.stringify(env.calls.btnText));
  check('退出后状态归 idle', env.calls.state.includes('idle'), JSON.stringify(env.calls.state));
}

console.log(failed === 0 ? 'FRONTEND_VOICE=PASS' : 'FRONTEND_VOICE=FAIL(' + failed + ')');
process.exit(failed === 0 ? 0 : 1);
