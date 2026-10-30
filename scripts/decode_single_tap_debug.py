"""Analyze a .ksdbg dump from iphone_live_demo.html's "Download this round (debug)" button.
Unlike decode_debug_session.py (which needs a keyLog of keydown timestamps), this replays
classifySingleTap()'s actual logic: look at the last TAIL_S seconds, find the single loudest
short-time-energy peak with no anchor at all, classify it. Also prints the energy profile around
the chosen peak so it's possible to tell "found the tap, model got it wrong" apart from "locked
onto the wrong sound entirely" (background noise, handling noise, etc.).
Usage: python scripts/decode_single_tap_debug.py path/to/iphone_round_XXXX.ksdbg
"""
import sys, json, struct, numpy as np, onnxruntime as ort

BEFORE, AFTER, TAIL_S = 2400, 12000, 6

def load_session(path):
    data = open(path, "rb").read()
    assert data[:5] == b"KSDBG", "not a .ksdbg file"
    (hlen,) = struct.unpack_from("<I", data, 5)
    header = json.loads(data[9:9 + hlen].decode("utf-8"))
    pcm = np.frombuffer(data[9 + hlen:], dtype="<i2").astype(np.float32) / 32767.0
    return header, pcm

def resample(x, src_sr, tgt_sr):
    if src_sr == tgt_sr: return x
    import scipy.signal as sig
    from math import gcd
    g = gcd(src_sr, tgt_sr)
    return sig.resample_poly(x, tgt_sr // g, src_sr // g).astype(np.float32)

def short_time_energy(buf, win=256, step=8):
    n = (len(buf) - win) // step
    e = np.empty(n, dtype=np.float64)
    for k in range(n):
        i = k * step
        e[k] = np.sum(np.abs(buf[i:i+win]))
    return e, step, win

def mel_features(x, melfb, n_fft=1536, hop=300, win_len=1024, n_mels=64):
    pad = n_fft // 2
    padded = np.pad(x, pad, mode="reflect")
    win = np.zeros(n_fft, dtype=np.float64)
    left = (n_fft - win_len) // 2
    win[left:left+win_len] = 0.5 - 0.5*np.cos(2*np.pi*np.arange(win_len)/win_len)
    n_frames = 1 + (len(padded) - n_fft) // hop
    out = np.zeros((n_mels, n_frames), dtype=np.float32)
    fb = melfb.reshape(n_mels, -1)
    for t in range(n_frames):
        frame = padded[t*hop:t*hop+n_fft] * win
        spec = np.fft.rfft(frame, n=n_fft)
        power = (spec.real**2 + spec.imag**2).astype(np.float64)
        out[:, t] = np.log2(fb @ power + 1e-10)
    return out, n_frames

def main():
    path = sys.argv[1]
    header, full = load_session(path)
    micSr, trainSr = header["sr"], header["trainSr"]
    print(f"session: micSr={micSr} trainSr={trainSr} set={header['set']} total_buf_sec={len(full)/micSr:.2f}")

    tail_len = min(len(full), round(TAIL_S * micSr))
    buf = full[len(full) - tail_len:]
    print(f"analyzing last {len(buf)/micSr:.2f}s (TAIL_S={TAIL_S})")
    resampled = resample(buf, micSr, trainSr)

    e, step, win = short_time_energy(resampled)
    order = np.argsort(-e)
    print("\ntop 8 energy peaks (sample index @ trainSr, time, relative energy):")
    top_vals = e[order[0]]
    seen = []
    for idx in order:
        t = (idx*step + win//2)
        if any(abs(t-s) < win for s in seen): continue
        seen.append(t)
        print(f"  idx={t:7d}  t={t/trainSr:.3f}s  energy={e[idx]:.4f}  ({100*e[idx]/top_vals:.0f}% of top)")
        if len(seen) >= 8: break

    onset = seen[0]
    print(f"\nglobal peak chosen: t={onset/trainSr:.3f}s, energy {e[order[0]]:.4f}")
    ambiguous = len(seen) > 1 and e[order[1]] if False else None

    norm = json.load(open(f"model/{header['set']}/norm.json"))
    melfb = np.array(json.load(open(f"model/{header['set']}/melfb.json"))["melfb"], dtype=np.float32)
    keys = norm["keys"]
    sess = ort.InferenceSession(f"model/{header['set']}/model.onnx")

    windowed = np.zeros(BEFORE+AFTER, dtype=np.float32)
    for i in range(BEFORE+AFTER):
        si = onset - BEFORE + i
        if 0 <= si < len(resampled): windowed[i] = resampled[si]
    print(f"window stats: max={np.abs(windowed).max():.4f} mean={np.abs(windowed).mean():.4f}")

    mel, nFrames = mel_features(windowed, melfb)
    mel = mel[:, :48]
    if mel.shape[1] < 48: mel = np.pad(mel, ((0,0),(0,48-mel.shape[1])))
    inp = ((mel - norm["mu"]) / norm["sd"]).astype(np.float32)[None, None, :, :]
    logits = sess.run(None, {"mel": inp})[0][0]
    p = np.exp(logits - logits.max()); p /= p.sum()
    ranked = np.argsort(-p)
    print("\ntop 8 guesses:")
    for i in ranked[:8]:
        print(f"  {keys[i]!r:4} {p[i]*100:.1f}%")

if __name__ == "__main__":
    main()
