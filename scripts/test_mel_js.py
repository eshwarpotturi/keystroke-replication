"""Cross-check scripts/mel.js against mel_fn() in train.py, numerically.
Generates a synthetic signal, runs it through the real Python feature pipeline, writes the
inputs/expected output to a JSON fixture, then runs mel.js under Node on the same input and
diffs the two. This is the only way to catch a subtle STFT/window/padding mismatch before it
silently produces garbage predictions on live audio.
Usage: python scripts/test_mel_js.py
"""
import json, subprocess, sys, numpy as np, torch, librosa

N_FFT, HOP, WIN, N_MELS, SR = 1536, 300, 1024, 64, 44100

def mel_fn(sr):
    fb = torch.tensor(librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=100, fmax=2500), dtype=torch.float32)
    win = torch.hann_window(WIN)
    def f(x):
        S = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=win, return_complex=True, center=True).abs()**2
        M = torch.matmul(fb, S)
        return torch.log2(M + 1e-10), fb
    return f

rng = np.random.RandomState(0)
# a synthetic "keystroke-ish" signal: short decaying click + noise, same length as a training window
T = 14400
t = np.arange(T) / SR
sig = (np.exp(-t * 40) * np.sin(2 * np.pi * 1800 * t) * 0.8 + rng.randn(T) * 0.02).astype(np.float32)

f = mel_fn(SR)
M, fb = f(torch.tensor(sig).unsqueeze(0))
M = M[0].numpy()  # (64, frames)
print("python mel shape", M.shape, "sample values", M[0, :5])

fixture = {
    "sr": SR,
    "signal": sig.tolist(),
    "melfb": fb.numpy().astype(np.float64).flatten().tolist(),  # row-major (64, 769)
    "expected_mel": M.astype(np.float64).tolist(),  # (64, frames)
    "n_frames": int(M.shape[1]),
}
open("/tmp/mel_fixture.json", "w").write(json.dumps(fixture))
print("wrote /tmp/mel_fixture.json, frames=", M.shape[1])

# run the JS side under node and diff
node_script = r"""
import { readFileSync, writeFileSync } from "fs";
import { logMel } from "/tmp/keystroke-replication/scripts/mel.js";
const fx = JSON.parse(readFileSync("/tmp/mel_fixture.json", "utf8"));
const x = Float64Array.from(fx.signal);
const melfb = Float32Array.from(fx.melfb);
const { mel, nFrames } = logMel(x, melfb);
writeFileSync("/tmp/mel_js_out.json", JSON.stringify({ mel: Array.from(mel), nFrames }));
"""
open("/tmp/run_mel.mjs", "w").write(node_script)
r = subprocess.run(["node", "/tmp/run_mel.mjs"], capture_output=True, text=True)
if r.returncode != 0:
    print("NODE FAILED:\n", r.stdout, r.stderr); sys.exit(1)

js = json.load(open("/tmp/mel_js_out.json"))
js_mel = np.array(js["mel"]).reshape(N_MELS, js["nFrames"])
print("js mel shape", js_mel.shape)

py_mel = np.array(fixture["expected_mel"])
n = min(py_mel.shape[1], js_mel.shape[1])
diff = np.abs(py_mel[:, :n] - js_mel[:, :n])
print(f"frames compared: {n} (py {py_mel.shape[1]}, js {js_mel.shape[1]})")
print(f"max abs diff: {diff.max():.6f}  mean abs diff: {diff.mean():.6f}")
print(f"py range [{py_mel.min():.3f},{py_mel.max():.3f}]  js range [{js_mel.min():.3f},{js_mel.max():.3f}]")
ok = diff.max() < 0.01 and py_mel.shape[1] == js_mel.shape[1]
print("MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
