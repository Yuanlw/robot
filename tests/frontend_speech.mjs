// 前端语音播放行为测试（node 直接跑 web.html 里的 scheduleSpeech，不靠人耳）。
// 回归 2026-11-14 静默无声事故：后端不返回时前端曾直接 return，既不提示也不出声。
// 用法：node tests/frontend_speech.mjs   （由 tests/test_frontend_speech.py 接入 unittest）
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const html = fs.readFileSync(path.join(root, 'app', 'web.html'), 'utf8');
const start = html.indexOf('function scheduleSpeech');
if (start < 0) { console.error('FAIL: web.html 里找不到 scheduleSpeech'); process.exit(1); }
const body = html.slice(start, html.indexOf('\n}', start) + 2);

function makeEnv({ fetchImpl }) {
  const log = [];
  const timers = [];
  globalThis.setTimeout = (fn, ms) => { const t = { fn, ms }; timers.push(t); return t; };
  globalThis.clearTimeout = (t) => { if (t) t.cleared = true; };
  globalThis.$ = () => ({ get checked() { return true; }, set textContent(v) { log.push(String(v)); } });
  globalThis.speechEpoch = 0;
  globalThis.speechRequests = new Set();
  globalThis.speechChain = Promise.resolve();
  globalThis.playbackPending = 0;
  globalThis.robotState = 'idle';
  globalThis.setRobotState = () => {};
  globalThis.firstAudioLogged = true;
  globalThis.turnStartedAt = 0;
  globalThis.currentAudio = null;
  globalThis.releasePlayback = null;
  globalThis.browserSpeak = async (text) => { log.push('BROWSER_SPEAK: ' + text); };
  globalThis.URL.createObjectURL = () => 'blob:stub';
  globalThis.URL.revokeObjectURL = () => {};
  globalThis.Audio = class {
    play() { queueMicrotask(() => this.onended && this.onended()); return Promise.resolve(); }
    pause() {}
  };
  globalThis.fetch = fetchImpl;
  const scheduleSpeech = eval('(' + body + ')');
  const flush = async () => {
    for (let i = 0; i < 5; i++) await new Promise((r) => setImmediate(r));
    await globalThis.speechChain.catch(() => {});
  };
  const fireTimers = () => { for (const t of timers) if (!t.cleared && t.fn) t.fn(); };
  return { log, timers, scheduleSpeech, flush, fireTimers };
}

const neverFetch = (_url, { signal }) => new Promise((_res, rej) => {
  signal.addEventListener('abort', () => { const e = new Error('aborted'); e.name = 'AbortError'; rej(e); });
});
const okFetch = async () => ({ ok: true, blob: async () => ({ stub: true }) });

let failed = 0;
function check(name, cond, detail) {
  console.log((cond ? 'PASS: ' : 'FAIL: ') + name + (cond ? '' : ' —— ' + detail));
  if (!cond) failed++;
}

// 1) 后端永不返回（复现 61 秒下载卡死）：超时后必须改用浏览器音色并提示原因
{
  const env = makeEnv({ fetchImpl: neverFetch });
  env.scheduleSpeech('你好，我在这里。', 0);
  await new Promise((r) => setImmediate(r));
  env.fireTimers();
  await env.flush();
  check('超时后改用浏览器音色朗读', env.log.some((l) => l.startsWith('BROWSER_SPEAK')), JSON.stringify(env.log));
  check('超时原因有提示', env.log.some((l) => l.includes('超时') && l.includes('浏览器中文音色')), JSON.stringify(env.log));
  const waits = env.timers.map((t) => t.ms);
  check('两段超时窗口都不超过前端预取上限 20s', waits.includes(20000) && waits.includes(15000) && waits.every((w) => w <= 20000), JSON.stringify(waits));
}

// 2) 用户主动停止（stopVoice：epoch++ 并 abort）：必须保持静默
{
  const env = makeEnv({ fetchImpl: neverFetch });
  env.scheduleSpeech('这句会被用户取消', 0);
  globalThis.speechEpoch = 1;
  for (const ctrl of globalThis.speechRequests) ctrl.abort();
  await env.flush();
  check('主动取消保持静默', !env.log.some((l) => l.startsWith('BROWSER_SPEAK')), JSON.stringify(env.log));
}

// 3) 正常路径：本地音频直接播放，不走浏览器兜底
{
  const env = makeEnv({ fetchImpl: okFetch });
  env.scheduleSpeech('正常播放的一句。', 0);
  await env.flush();
  check('正常路径播放本地音频', env.log.some((l) => l.includes('正在播放中文语音')) && env.log.some((l) => l.includes('语音播放完毕')), JSON.stringify(env.log));
  check('正常路径不触发浏览器兜底', !env.log.some((l) => l.startsWith('BROWSER_SPEAK')), JSON.stringify(env.log));
}

console.log(failed === 0 ? 'FRONTEND_SPEECH=PASS' : 'FRONTEND_SPEECH=FAIL(' + failed + ')');
process.exit(failed === 0 ? 0 : 1);
