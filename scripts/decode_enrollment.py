"""Decode a .ksd file from scripts/enroll.html into data/<name>.npz, the format train.py expects
(X: float32 (n, strokeLen) raw audio windows, y: int labels, pos: int press-index-within-key, sr: int).
Usage: python scripts/decode_enrollment.py path/to/enrollment_XXXX.ksd [name=office]
"""
import sys, json, struct, numpy as np

def decode(path):
    data = open(path, "rb").read()
    assert data[:4] == b"KSTR", "not a .ksd file (bad magic)"
    (hlen,) = struct.unpack_from("<I", data, 4)
    header = json.loads(data[8:8 + hlen].decode("utf-8"))
    pcm_bytes = data[8 + hlen:]
    pcm = np.frombuffer(pcm_bytes, dtype="<i2")
    n, L = header["n"], header["strokeLen"]
    assert pcm.shape[0] == n * L, f"size mismatch: expected {n*L} int16 samples, got {pcm.shape[0]}"
    X = (pcm.reshape(n, L).astype(np.float32)) / 32767.0
    keys = header["keys"]
    y = np.array([keys.index(c) for c in header["labels"]], dtype=np.int64)
    pos = np.array(header["pos"], dtype=np.int64)
    sr = header["sr"]
    return X, y, pos, sr, keys

if __name__ == "__main__":
    path = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "office"
    X, y, pos, sr, keys = decode(path)
    print(f"decoded {len(y)} strokes, sr={sr}, keys seen: {sorted(set(y))} (of {len(keys)})")
    counts = np.bincount(y, minlength=len(keys))
    short = [keys[i] for i, c in enumerate(counts) if c < 10]
    if short:
        print(f"WARNING: these keys have fewer than 10 presses, may hurt accuracy: {short}")
    import os
    os.makedirs("data", exist_ok=True)
    np.savez(f"data/{name}.npz", X=X, y=y, pos=pos, sr=sr)
    print(f"wrote data/{name}.npz")
