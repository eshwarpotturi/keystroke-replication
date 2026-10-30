"""Analyze a .ksdbg dump from iphone_live_demo.html's 3-tap "Download this round (debug)" button.
Replays classifyMultiTap(n)'s exact logic: last (4n+2) seconds, n separated peaks by energy
(0.3s min gap), classify each, average. Prints each tap's individual verdict plus the average, so
a near-miss averaging problem is visible separately from "didn't find 3 real taps at all".
Usage: python scripts/decode_multitap_debug.py path/to/iphone_round_XXXX.ksdbg [n=3]
"""
import sys, json, struct, numpy as np, onnxruntime as ort

BEFORE, AFTER = 2400, 12000

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
    n = max(0, (len(buf) - win) // step)
    e = np.empty(n, dtype=np.float64)
    for k in range(n):
        i = k * step
        e[k] = np.sum(np.abs(buf[i:i+win]))
    return e, step, win

def find_top_n_peaks(buf, n, min_sep):
    e, step, win = short_time_energy(buf)
    order = np.argsort(-e)
    picked = []
    for idx in order:
        t = idx*step + win//2
        if len(picked) >= n: break
        if all(abs(t-p) > min_sep for p in picked): picked.append(t)
    picked.sort()
    return picked, e, order, step, win

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
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    header, full = load_session(path)
    micSr, trainSr = header["sr"], header["trainSr"]
    print(f"session: micSr={micSr} trainSr={trainSr} set={header['set']} total_buf_sec={len(full)/micSr:.2f}")

    TAIL_S = 4*n + 2
    tail_len = min(len(full), round(TAIL_S * micSr))
    buf = full[len(full) - tail_len:]
    print(f"analyzing last {len(buf)/micSr:.2f}s (TAIL_S={TAIL_S} for n={n})")
    resampled = resample(buf, micSr, trainSr)

    min_sep = round(0.3 * trainSr)
    peaks, e, order, step, win = find_top_n_peaks(resampled, n, min_sep)
    print(f"\nfound {len(peaks)} of {n} separated peaks, at times: {[round(p/trainSr,3) for p in peaks]}")
    if len(peaks) < n:
        print("FEWER THAN REQUESTED — this round would have shown the retry message, not a guess.")
        return

    norm = json.load(open(f"model/{header['set']}/norm.json"))
    melfb = np.array(json.load(open(f"model/{header['set']}/melfb.json"))["melfb"], dtype=np.float32)
    keys = norm["keys"]
    sess = ort.InferenceSession(f"model/{header['set']}/model.onnx")

    probs = []
    for i, onset in enumerate(peaks):
        windowed = np.zeros(BEFORE+AFTER, dtype=np.float32)
        for j in range(BEFORE+AFTER):
            si = onset - BEFORE + j
            if 0 <= si < len(resampled): windowed[j] = resampled[si]
        mel, nFrames = mel_features(windowed, melfb)
        mel = mel[:, :48]
        if mel.shape[1] < 48: mel = np.pad(mel, ((0,0),(0,48-mel.shape[1])))
        inp = ((mel - norm["mu"]) / norm["sd"]).astype(np.float32)[None, None, :, :]
        logits = sess.run(None, {"mel": inp})[0][0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        probs.append(p)
        top3 = np.argsort(-p)[:3]
        print(f"  tap {i+1} (t={onset/trainSr:.3f}s, wmax={np.abs(windowed).max():.4f}): " +
              ", ".join(f"{keys[k]!r} {p[k]*100:.1f}%" for k in top3))

    avg = np.mean(probs, axis=0)
    ranked = np.argsort(-avg)
    print("\naveraged top 8:")
    for i in ranked[:8]:
        print(f"  {keys[i]!r:4} {avg[i]*100:.1f}%")

if __name__ == "__main__":
    main()
