class PCMCapture extends AudioWorkletProcessor {
  constructor(opts) {
    super();
    const po = (opts && opts.processorOptions) || {};
    this.target = po.targetRate || 16000;
    this.ratio = sampleRate / this.target;   // sampleRate 是 worklet 作用域全局
    this.acc = new Float32Array(0);
    // 每攒够 ~200ms(16k) 就发一帧，兼顾延迟和 WS 开销
    this.need = Math.ceil(this.ratio * 3200);
    this.levelEvery = Math.ceil(sampleRate * 0.15);   // 每 ~150ms 报一次电平
    this.levelAcc = 0;
    this.levelPeak = 0;
    // 松手时主线程会发 'flush'，把不足一帧的尾巴也交出去
    this.port.onmessage = (e) => {
      if (e.data === 'flush') this.emit(true);
    };
  }

  /** 把 acc 里已有的采样重采样成 16k Int16 并发出。force=true 时不管够不够一帧。 */
  emit(force) {
    if (this.acc.length === 0) return;
    if (!force && this.acc.length < this.need) return;

    const outLen = Math.floor(this.acc.length / this.ratio);
    if (outLen <= 0) {
      if (force) this.acc = new Float32Array(0);
      return;
    }
    const out = new Int16Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * this.ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, this.acc.length - 1);
      const f = pos - i0;
      let s = this.acc[i0] * (1 - f) + this.acc[i1] * f;   // 线性插值重采样
      s = Math.max(-1, Math.min(1, s));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    // 只丢弃已消费的余数（< ratio 个采样点），不要整块清空
    const consumed = Math.min(Math.ceil(outLen * this.ratio), this.acc.length);
    this.acc = this.acc.slice(consumed);
    this.port.postMessage(out.buffer, [out.buffer]);
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch || ch.length === 0) return true;

    // 电平统计：全零说明麦克风根本没在采集（权限/设备/音频图被拆），
    // 这种情况必须当场可见，否则静音会被送进 ASR 变成幻觉文本。
    for (let i = 0; i < ch.length; i++) {
      const a = ch[i] < 0 ? -ch[i] : ch[i];
      if (a > this.levelPeak) this.levelPeak = a;
      this.levelAcc += ch[i] * ch[i];
    }
    this.levelN = (this.levelN || 0) + ch.length;
    if (this.levelN >= this.levelEvery) {
      this.port.postMessage({
        level: 1,
        peak: this.levelPeak,
        rms: Math.sqrt(this.levelAcc / this.levelN),
      });
      this.levelAcc = 0; this.levelPeak = 0; this.levelN = 0;
    }

    const merged = new Float32Array(this.acc.length + ch.length);
    merged.set(this.acc);
    merged.set(ch, this.acc.length);
    this.acc = merged;

    this.emit(false);
    return true;
  }
}

registerProcessor('pcm-capture', PCMCapture);
