// Live microphone demo: type a warm-up passage with live per-letter guesses, then type a hidden
// password and reveal what the model heard. Runs entirely client-side (mic audio never leaves
// the page). Requires model/<set>/model.onnx + melfb.json + norm.json to exist (see model/README.md);
// until then the panel says so plainly instead of pretending to work.
import { logMel, prepModelInput, N_FRAMES_USED } from "./mel.js";

const SETS = ["phone", "zoom", "office", "iphone"]; // which trained, same-device model to load
// Window shape as RAW SAMPLE COUNTS at the training sample rate — must match scripts/enroll.html's
// BEFORE/AFTER exactly (both are literally 2400/12000), not a time duration computed against some
// assumed sample rate. The office/phone/zoom models were each trained at their own sr (48000, 44100,
// 32000...); treating BEFORE/AFTER as "2400/44100 of a second" silently resizes and misaligns the
// window for any model not trained at 44100 — which is exactly what office (48000) hit.
const BEFORE = 2400, AFTER = 12000;
const SEARCH_S = 0.15; // how far around the estimated keydown time to search for the real onset

let live = {
  set: null, sess: null, melfb: null, norm: null,
  ctx: null, stream: null, proc: null, gain: null,
  chunks: [], t0Perf: 0, sr: 0,
  keyLog: [], // {ch, tPerf}
  classified: [], // cache, parallel to keyLog — classifyAll() only processes new entries
  active: false,
  marginMs: 1000, // placeholder until init() computes the real value from the loaded model's sr
};

async function loadModel(set) {
  const base = `model/${set}`;
  const [melfbR, normR] = await Promise.all([fetch(`${base}/melfb.json`), fetch(`${base}/norm.json`)]);
  if (!melfbR.ok || !normR.ok) throw new Error("no exported model for " + set);
  const melfbJ = await melfbR.json(), norm = await normR.json();
  const sess = await ort.InferenceSession.create(`${base}/model.onnx`, { executionProviders: ["wasm"] });
  return { sess, melfb: Float32Array.from(melfbJ.melfb), norm };
}

async function startMic() {
  live.ctx = new (window.AudioContext || window.webkitAudioContext)();
  live.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  const src = live.ctx.createMediaStreamSource(live.stream);
  // ScriptProcessorNode is deprecated but simplest for a short-lived demo page; buffer size 4096.
  live.proc = live.ctx.createScriptProcessor(4096, 1, 1);
  live.gain = live.ctx.createGain(); live.gain.gain.value = 0; // keep graph alive, no audible loopback
  live.sr = live.ctx.sampleRate;
  live.chunks = []; live.keyLog = [];
  // anchor: perf time that corresponds to ctx.currentTime===0 for this context
  live.t0Perf = performance.now() - live.ctx.currentTime * 1000;
  live.proc.onaudioprocess = (e) => { if (live.active) live.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
  src.connect(live.proc); live.proc.connect(live.gain); live.gain.connect(live.ctx.destination);
  live.active = true;
}

function stopMic() {
  live.active = false;
  try { live.proc && live.proc.disconnect(); live.gain && live.gain.disconnect(); } catch {}
  try { live.stream && live.stream.getTracks().forEach(t => t.stop()); } catch {}
}

function fullBuffer() {
  const n = live.chunks.reduce((s, c) => s + c.length, 0);
  const out = new Float32Array(n); let o = 0;
  for (const c of live.chunks) { out.set(c, o); o += c.length; }
  return out;
}

// Energy-peak onset search (same idea as scripts/isolate.py), restricted to a small window
// around the keydown-estimated sample index instead of scanning the whole recording.
function findOnset(buf, sr, centerIdx, searchS) {
  const half = Math.round(searchS * sr);
  const lo = Math.max(0, centerIdx - half), hi = Math.min(buf.length, centerIdx + half);
  const win = 256; // short-time energy window
  let best = centerIdx, bestE = -1;
  for (let i = lo; i + win < hi; i += 8) {
    let e = 0; for (let j = 0; j < win; j++) e += Math.abs(buf[i + j]);
    if (e > bestE) { bestE = e; best = i + (win >> 1); }
  }
  return best;
}

// Resample a short slice to targetSr using OfflineAudioContext (cheap: slices are ~0.3s).
async function resampleSlice(float32, srcSr, targetSr) {
  if (srcSr === targetSr) return float32;
  const off = new OfflineAudioContext(1, Math.ceil(float32.length * targetSr / srcSr), targetSr);
  const buf = off.createBuffer(1, float32.length, srcSr);
  buf.copyToChannel(float32, 0);
  const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0).slice();
}

// Run the model on a (trainSr, BEFORE+AFTER samples) window and return the softmax verdict.
async function classifyWindow(windowed) {
  const { mel, nFrames } = logMel(windowed, live.melfb);
  const input = prepModelInput(mel, nFrames, live.norm.mu, live.norm.sd);
  const tensor = new ort.Tensor("float32", input, [1, 1, 64, N_FRAMES_USED]);
  const out = await live.sess.run({ mel: tensor });
  const logits = out.logits.data;
  // softmax
  let mx = -Infinity; for (const v of logits) mx = Math.max(mx, v);
  let sum = 0; const p = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) { p[i] = Math.exp(logits[i] - mx); sum += p[i]; }
  for (let i = 0; i < p.length; i++) p[i] /= sum;
  const keys = live.norm.keys;
  let best = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
  const ranked = [...p.keys()].sort((a, b) => p[b] - p[a]).map(i => keys[i]);
  return { guess: keys[best], conf: p[best], top3: ranked.slice(0, 3), top5: ranked.slice(0, 5) };
}

// Classify one keystroke: raw mic buffer, its sample rate, and the estimated sample index of the press.
async function classifyStroke(buf, micSr, idxMic) {
  const trainSr = live.norm.sr;
  // BEFORE/AFTER are sample counts at trainSr; convert to a time duration, then to mic-sr samples,
  // to size the raw (pre-resample) slice we pull out of the live buffer.
  const beforeSec = BEFORE / trainSr, afterSec = AFTER / trainSr;
  const needBefore = Math.round((SEARCH_S + beforeSec) * micSr), needAfter = Math.round((SEARCH_S + afterSec) * micSr);
  const lo = Math.max(0, idxMic - needBefore), hi = Math.min(buf.length, idxMic + needAfter);
  const raw = buf.slice(lo, hi);
  const resampled = await resampleSlice(raw, micSr, trainSr);
  const localCenter = Math.round((idxMic - lo) * trainSr / micSr);
  const onset = findOnset(resampled, trainSr, localCenter, SEARCH_S);
  // from here on, BEFORE/AFTER are used as raw sample counts in the resampled (trainSr) buffer —
  // identical in kind to what scripts/enroll.html used when it built the training windows.
  const windowed = new Float64Array(BEFORE + AFTER);
  for (let i = 0; i < BEFORE + AFTER; i++) {
    const si = onset - BEFORE + i;
    windowed[i] = si >= 0 && si < resampled.length ? resampled[si] : 0;
  }
  return classifyWindow(windowed);
}

// Find the single loudest short-time-energy peak anywhere in the buffer — no keydown anchor needed
// at all. Used for the no-sync "tap once, reveal" flow (two devices with no way to coordinate timing).
function findGlobalPeak(buf) {
  const win = 256, step = 8;
  let best = win >> 1, bestE = -1;
  for (let i = 0; i + win < buf.length; i += step) {
    let e = 0; for (let j = 0; j < win; j++) e += Math.abs(buf[i + j]);
    if (e > bestE) { bestE = e; best = i + (win >> 1); }
  }
  return best;
}

// Public API used by the page's UI wiring (added in template.html).
window.KeystrokeLive = {
  async init(set) {
    const m = await loadModel(set);
    live.set = set; live.sess = m.sess; live.melfb = m.melfb; live.norm = m.norm;
    // a keystroke needs (SEARCH_S + AFTER/trainSr) of audio recorded *after* it before it can be
    // classified; this depends on the loaded model's own sample rate, so it's computed here rather
    // than assumed at module load — the same mistake that broke BEFORE_S/AFTER_S above.
    live.marginMs = Math.ceil((SEARCH_S + AFTER / m.norm.sr) * 1000) + 250;
    return m.norm;
  },
  startMic,
  stopMic,
  onKeydown(ch) { if (live.active) live.keyLog.push({ ch, tPerf: performance.now() }); },
  // classify only keystrokes not yet classified (cached in live.classified), append, return the
  // full cumulative array [{ch,guess,conf,top3}] so callers can always read results[results.length-1].
  async classifyAll() {
    const buf = fullBuffer(), micSr = live.sr, now = performance.now();
    for (let i = live.classified.length; i < live.keyLog.length; i++) {
      const k = live.keyLog[i];
      if (now - k.tPerf < live.marginMs) break; // not enough trailing audio captured yet — try again next call
      const idx = Math.round(((k.tPerf - live.t0Perf) / 1000) * micSr);
      try { live.classified.push({ ch: k.ch, ...(await classifyStroke(buf, micSr, idx)) }); }
      catch (e) { live.classified.push({ ch: k.ch, guess: "?", conf: 0, top3: [] }); }
    }
    return live.classified;
  },
  reset() { live.chunks = []; live.keyLog = []; live.classified = []; },
  // Like reset() but doesn't touch the mic session — for "tap again" between rounds of the
  // no-sync single-tap demo, where re-requesting the mic each round would be disruptive.
  clearBuffer() { live.chunks = []; live.keyLog = []; live.classified = []; },
  // No-sync classification: finds the single loudest tap anywhere in the buffer recorded since
  // the last clearBuffer()/reset(), and classifies it. Works for exactly one tap in the buffer —
  // call clearBuffer() before each new tap. Doesn't need any keydown signal or timing coordination
  // with a separate device, unlike classifyAll()/onKeydown().
  async classifySingleTap() {
    const buf = fullBuffer(), micSr = live.sr, trainSr = live.norm.sr;
    const resampled = await resampleSlice(buf, micSr, trainSr);
    const onset = findGlobalPeak(resampled);
    const windowed = new Float64Array(BEFORE + AFTER);
    for (let i = 0; i < BEFORE + AFTER; i++) {
      const si = onset - BEFORE + i;
      windowed[i] = si >= 0 && si < resampled.length ? resampled[si] : 0;
    }
    return classifyWindow(windowed);
  },
  hasModel(set) { return SETS.includes(set); },
  keyLogLength() { return live.keyLog.length; },
  // Package the raw continuous buffer + keydown log for offline debugging (not used by the normal
  // UI flow). Triggers a download; the file can be analyzed with scripts/decode_debug_session.py.
  dumpSession(filename) {
    const buf = fullBuffer();
    const header = JSON.stringify({
      sr: live.sr, set: live.set, trainSr: live.norm ? live.norm.sr : null,
      marginMs: live.marginMs,
      keyLog: live.keyLog.map(k => ({ ch: k.ch, tRelMs: k.tPerf - live.t0Perf })),
    });
    const headerBytes = new TextEncoder().encode(header);
    const magic = new TextEncoder().encode("KSDBG");
    const lenBuf = new Uint32Array([headerBytes.length]);
    const pcm = new Int16Array(buf.length);
    for (let i = 0; i < buf.length; i++) pcm[i] = Math.max(-32768, Math.min(32767, Math.round(buf[i] * 32767)));
    const blob = new Blob([magic, new Uint8Array(lenBuf.buffer), headerBytes, new Uint8Array(pcm.buffer)], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = filename || `debug_session_${Date.now()}.ksdbg`; a.click();
  },
};
