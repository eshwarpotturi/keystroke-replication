"""Decode a .ksiph file from scripts/enroll_iphone.html into data/<name>.npz, the format
train.py expects (X, y, pos, sr). Unlike decode_enrollment.py (which has keydown timestamps to
anchor each stroke), this has to find taps "blind" -- purely from the audio -- because the
recording device has no visibility into what's tapped on the separate iPhone. Reuses the exact
isolator from scripts/isolate.py (the method the original phone/Zoom datasets were built with),
run separately within each key's segment, tuning the energy threshold until exactly `perKey`
peaks are found, same as isolate.py does per per-key wav file.
Usage: python scripts/decode_iphone_enrollment.py path/to/iphone_session_XXXX.ksiph [name=iphone]
"""
import sys, json, struct, time, numpy as np, librosa

KEYS = "1234567890qwertyuiopasdfghjklzxcvbnm"
SIZE, SCAN, BEFORE, AFTER = 48, 24, 2400, 12000

def log(*a): print(time.strftime("%Y-%m-%dT%H:%M:%S"), *a, flush=True)

def load_session(path):
    data = open(path, "rb").read()
    assert data[:5] == b"KSIPH", "not a .ksiph file (bad magic)"
    (hlen,) = struct.unpack_from("<I", data, 5)
    header = json.loads(data[9:9 + hlen].decode("utf-8"))
    pcm = np.frombuffer(data[9 + hlen:], dtype="<i2").astype(np.float32) / 32767.0
    return header, pcm

def isolator(signal, sr, size, scan, before, after, threshold):
    """Verbatim port of scripts/isolate.py's isolator() -- same energy-threshold peak pick."""
    strokes = []
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    peaks = np.where(energy > threshold)[0]
    prev_end = sr * 0.1 * (-1)
    for p in peaks:
        ts = (p * scan) + size // 2
        if ts > prev_end + 0.1 * sr and ts - before >= 0 and ts + after <= len(signal):
            strokes.append(signal[ts - before:ts + after])
            prev_end = ts + after
    return strokes

def find_n_strokes(signal, sr, n, size=SIZE, scan=SCAN, before=BEFORE, after=AFTER):
    """Tune the threshold until exactly n strokes are found, same iterative approach isolate.py uses."""
    prom, step = 0.06, 0.005
    st = []
    for it in range(2000):
        st = isolator(signal, sr, size, scan, before, after, prom)
        if len(st) == n: break
        prom += step if len(st) > n else -step
        step *= 0.99
        if prom <= 0: break
    return st, prom

def main():
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "iphone"
    header, buf = load_session(path)
    sr, per_key, boundaries = header["sr"], header["perKey"], header["boundaries"]
    log(f"session: sr={sr} per_key={per_key} keys={len(boundaries)} total_sec={len(buf)/sr:.1f}")

    X, y, pos = [], [], []
    short = []
    start = 0
    for i, b in enumerate(boundaries):
        ch, end = b["ch"], b["endSample"]
        seg = buf[start:end]
        start = end
        strokes, thr = find_n_strokes(seg, sr, per_key)
        ki = KEYS.index(ch)
        for j, s in enumerate(strokes):
            X.append(s); y.append(ki); pos.append(j)
        log(f"key {ch!r:3} ({i+1}/{len(boundaries)}): found {len(strokes)}/{per_key} thr={thr:.4f} seg_len={len(seg)/sr:.2f}s")
        if len(strokes) < per_key:
            short.append((ch, len(strokes)))

    if short:
        log(f"WARNING: under target count for: {short}")
    X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int64); pos = np.array(pos, dtype=np.int64)
    log(f"total strokes decoded: {len(y)} across {len(set(y.tolist()))} keys")
    import os
    os.makedirs("data", exist_ok=True)
    np.savez(f"data/{name}.npz", X=X, y=y, pos=pos, sr=sr)
    log(f"wrote data/{name}.npz")

if __name__ == "__main__":
    main()
