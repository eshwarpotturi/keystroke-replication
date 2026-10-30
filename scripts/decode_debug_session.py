"""Analyze a .ksdbg debug dump from the Live tab's "Download session (debug)" button.
Replays the exact same onset-search + windowing live.js uses, against the real exported model,
and reports per-keystroke diagnostics: inter-key gap, onset confidence, predicted vs typed, so a
real cause (overlap, bad onset, weak model) can be told apart from a guess.
Usage: python scripts/decode_debug_session.py path/to/debug_session_XXXX.ksdbg
"""
import sys, json, struct, numpy as np, onnxruntime as ort

BEFORE, AFTER, SEARCH_S = 2400, 12000, 0.15

def load_session(path):
    data = open(path, "rb").read()
    assert data[:5] == b"KSDBG", "not a .ksdbg file"
    (hlen,) = struct.unpack_from("<I", data, 5)
    header = json.loads(data[9:9 + hlen].decode("utf-8"))
    pcm = np.frombuffer(data[9 + hlen:], dtype="<i2").astype(np.float32) / 32767.0
    return header, pcm

def find_onset(buf, sr, center_idx, search_s):
    half = round(search_s * sr)
    lo, hi = max(0, center_idx - half), min(len(buf), center_idx + half)
    win = 256
    best, best_e = center_idx, -1
    i = lo
    while i + win < hi:
        e = np.sum(np.abs(buf[i:i+win]))
        if e > best_e: best_e, best = e, i + win // 2
        i += 8
    return best, best_e

def resample(x, src_sr, tgt_sr):
    if src_sr == tgt_sr: return x
    import scipy.signal as sig
    from math import gcd
    g = gcd(src_sr, tgt_sr)
    return sig.resample_poly(x, tgt_sr // g, src_sr // g).astype(np.float32)

def mel_features(x, melfb, n_fft=1536, hop=300, win_len=1024, n_mels=64):
    # must match train.py's mel_fn / scripts/mel.js exactly
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
    header, buf = load_session(path)
    micSr, trainSr, keyLog = header["sr"], header["trainSr"], header["keyLog"]
    print(f"session: micSr={micSr} trainSr={trainSr} set={header['set']} marginMs={header.get('marginMs')} n_keys={len(keyLog)} buf_sec={len(buf)/micSr:.2f}")

    norm = json.load(open(f"model/{header['set']}/norm.json"))
    melfb = np.array(json.load(open(f"model/{header['set']}/melfb.json"))["melfb"], dtype=np.float32)
    keys = norm["keys"]
    sess = ort.InferenceSession(f"model/{header['set']}/model.onnx")

    print("\ninter-key gaps (ms):", [round(keyLog[i]["tRelMs"]-keyLog[i-1]["tRelMs"]) for i in range(1,len(keyLog))])

    correct = 0
    for i, k in enumerate(keyLog):
        ch, tRelMs = k["ch"], k["tRelMs"]
        idx = round((tRelMs/1000) * micSr)
        beforeSec, afterSec = BEFORE/trainSr, AFTER/trainSr
        needBefore, needAfter = round((SEARCH_S+beforeSec)*micSr), round((SEARCH_S+afterSec)*micSr)
        lo, hi = max(0, idx-needBefore), min(len(buf), idx+needAfter)
        raw = buf[lo:hi]
        resampled = resample(raw, micSr, trainSr)
        localCenter = round((idx-lo) * trainSr / micSr)
        onset, onset_e = find_onset(resampled, trainSr, localCenter, SEARCH_S)
        windowed = np.zeros(BEFORE+AFTER, dtype=np.float32)
        for j in range(BEFORE+AFTER):
            si = onset - BEFORE + j
            if 0 <= si < len(resampled): windowed[j] = resampled[si]
        mel, nFrames = mel_features(windowed, melfb)
        mel = mel[:, :48]
        if mel.shape[1] < 48:
            mel = np.pad(mel, ((0,0),(0,48-mel.shape[1])))
        inp = ((mel - norm["mu"]) / norm["sd"]).astype(np.float32)[None, None, :, :]
        logits = sess.run(None, {"mel": inp})[0][0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        pred = keys[int(np.argmax(p))]
        ok = pred == ch
        correct += ok
        gap_before = "" if i == 0 else f" gap_before={round(tRelMs-keyLog[i-1]['tRelMs'])}ms"
        print(f"  [{i:3d}] typed={ch!r:4} pred={pred!r:4} conf={p.max():.2f} {'OK' if ok else 'WRONG'}  onset_local={onset-localCenter:+d}samp  wmax={np.abs(windowed).max():.4f}{gap_before}")
    print(f"\n{correct} of {len(keyLog)} correct ({100*correct/len(keyLog):.1f}%)")

if __name__ == "__main__":
    main()
