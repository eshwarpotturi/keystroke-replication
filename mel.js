// Exact JS port of mel_fn() in train.py: torch.stft(n_fft=1536, hop=300, win_length=1024,
// window=hann(1024,periodic), center=True, pad_mode="reflect") -> |.|^2 -> mel filterbank -> log2.
// Runs as a plain ES module in the browser, and under Node for the cross-check in test_mel.py.
//
// Matching notes (these are the parts that are easy to get subtly wrong):
// - torch.hann_window(1024) is periodic: w[n] = 0.5 - 0.5*cos(2*pi*n/N), n=0..N-1 (divide by N, not N-1).
// - win_length(1024) < n_fft(1536): torch center-pads the window with zeros to n_fft, (1536-1024)/2=256 zeros each side.
// - center=True pads the signal with n_fft//2=768 samples of *reflect* padding (no edge repeat) on each side.
// - frame length is n_fft (1536) directly; FFT size is n_fft, keep bins 0..n_fft/2 inclusive (769 bins).
// - mel filterbank values are NOT recomputed here: they're loaded from melfb_<name>.json, exported by
//   Python's librosa.filters.mel so the numbers are guaranteed identical, not just "close".

export const N_FFT = 1536, HOP = 300, WIN = 1024, N_MELS = 64, N_BINS = N_FFT / 2 + 1, N_FRAMES_USED = 48;

function hannPeriodic(N) {
  const w = new Float64Array(N);
  for (let n = 0; n < N; n++) w[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N);
  return w;
}

// torch pads a shorter window to n_fft by centering it with zeros.
function paddedWindow() {
  const w = hannPeriodic(WIN);
  const out = new Float64Array(N_FFT);
  const left = Math.floor((N_FFT - WIN) / 2);
  for (let n = 0; n < WIN; n++) out[left + n] = w[n];
  return out;
}

// numpy/torch "reflect" padding: no repeat of the edge sample.
// left pad = x[pad..1] reversed, i.e. x[1],x[2],...,x[pad] reversed -> x[pad],...,x[1]
function reflectPad(x, pad) {
  const T = x.length, out = new Float64Array(T + 2 * pad);
  for (let i = 0; i < pad; i++) out[i] = x[pad - i]; // x[pad] .. x[1]
  for (let i = 0; i < T; i++) out[pad + i] = x[i];
  for (let i = 0; i < pad; i++) out[pad + T + i] = x[T - 2 - i]; // x[T-2] .. x[T-1-pad]
  return out;
}

let COS = null, SIN = null; // (N_BINS x N_FFT), built once
function basis() {
  if (COS) return;
  COS = new Float32Array(N_BINS * N_FFT);
  SIN = new Float32Array(N_BINS * N_FFT);
  for (let k = 0; k < N_BINS; k++) {
    const w = (2 * Math.PI * k) / N_FFT;
    for (let n = 0; n < N_FFT; n++) {
      COS[k * N_FFT + n] = Math.cos(w * n);
      SIN[k * N_FFT + n] = -Math.sin(w * n); // e^{-j w n}
    }
  }
}

// one STFT frame -> power spectrum (N_BINS,), frame must already be windowed+padded to N_FFT
function framePower(frame) {
  basis();
  const pw = new Float32Array(N_BINS);
  for (let k = 0; k < N_BINS; k++) {
    let re = 0, im = 0;
    const base = k * N_FFT;
    for (let n = 0; n < N_FFT; n++) {
      const v = frame[n];
      re += COS[base + n] * v;
      im += SIN[base + n] * v;
    }
    pw[k] = re * re + im * im;
  }
  return pw;
}

// x: Float32Array/Float64Array of raw samples (already at the training sample rate).
// melfb: Float32Array length N_MELS*N_BINS, row-major (mel row, bin col) — from melfb_<name>.json.
// Returns {mel: Float32Array(N_MELS*nFrames), nFrames} of log2(mel power + 1e-10), UNNORMALIZED
// (matches Mte in train.py before the (M-mu)/sd normalization step).
export function logMel(x, melfb) {
  const win = paddedWindow();
  const pad = N_FFT >> 1; // 768
  const padded = reflectPad(x, pad);
  const nFrames = 1 + Math.floor((padded.length - N_FFT) / HOP);
  const framed = new Float64Array(N_FFT);
  const out = new Float32Array(N_MELS * nFrames);
  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP;
    for (let n = 0; n < N_FFT; n++) framed[n] = padded[start + n] * win[n];
    const pw = framePower(framed); // (N_BINS,)
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      const fbRow = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) s += melfb[fbRow + k] * pw[k];
      out[m * nFrames + t] = Math.log2(s + 1e-10);
    }
  }
  return { mel: out, nFrames };
}

// Crop/pad a (N_MELS, nFrames) mel array to exactly N_FRAMES_USED frames (feats() does M[...,:48]),
// then normalize with (x-mu)/sd, and return a flat Float32Array ready for the model (1,1,64,48).
export function prepModelInput(mel, nFrames, mu, sd) {
  const out = new Float32Array(N_MELS * N_FRAMES_USED);
  const useFrames = Math.min(N_FRAMES_USED, nFrames);
  for (let m = 0; m < N_MELS; m++) {
    for (let t = 0; t < useFrames; t++) out[m * N_FRAMES_USED + t] = (mel[m * nFrames + t] - mu) / sd;
    // if nFrames < 48 (short window), leave the rest at (0-mu)/sd implicitly via zero-fill below
  }
  if (nFrames < N_FRAMES_USED) {
    for (let m = 0; m < N_MELS; m++)
      for (let t = nFrames; t < N_FRAMES_USED; t++) out[m * N_FRAMES_USED + t] = (0 - mu) / sd;
  }
  return out;
}
