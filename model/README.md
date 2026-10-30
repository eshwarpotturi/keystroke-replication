# model/

Empty until a model is exported. The Live tab reads `model/<set>/model.onnx`,
`melfb.json` and `norm.json` at runtime; if they're missing it says so on the
page instead of pretending to work.

## Fastest path: record directly for the live demo (no raw dataset needed)

1. Open `scripts/enroll.html` (or the hosted copy, `enroll.html` at the repo
   root) on the exact laptop/mic you'll demo on. Press Start, allow the mic,
   and press each key shown ~20 times (10–15 min for all 36 keys).
2. It downloads a single `.ksd` file (a few MB) — short snippets around each
   press, nothing else.
3. Decode it and train:
   ```
   python scripts/decode_enrollment.py path/to/enrollment_XXXX.ksd office
   EXPORT=1 python scripts/train.py office office 0 150 random
   python scripts/build_report.py
   ```
   This produces `model/office/*` and the page's "This laptop" option in the
   Live tab starts working. Verified end to end (recorder → decode → train →
   export → in-browser inference) with a short synthetic run before asking
   you to spend the 10–15 minutes on the real one.

## iPhone keyboard taps, separate mic listening in

A different question from the office/phone/zoom models above: does an iPhone's
on-screen keyboard leak anything? It plays the same synthesized click for
every key (not a mechanical key with per-key resonance), so this isn't a
given — it might turn out keys are indistinguishable, which is itself a real
finding, not a failed demo.

Recording works differently here: the listening device (a laptop) has no way
to know what's tapped on a separate iPhone, so there's no keydown signal to
anchor each press to — unlike `enroll.html`. Instead it records continuously
and finds each tap "blind", from the audio energy alone, exactly how
`scripts/isolate.py` builds the original phone/Zoom datasets (one key at a
time, N presses, found by tuning an energy threshold until N peaks turn up).

1. Open `scripts/enroll_iphone.html` (or the hosted `enroll_iphone.html`) on
   the **listening device**. Keep the iPhone's keyboard click sound on
   (Settings → Sounds & Haptics → Keyboard Feedback), hold or place it near
   the listening device's mic like the original phone/Zoom setup.
2. For each key shown, tap it on the iPhone's on-screen keyboard the stated
   number of times, then press **Next key**. Switch the iPhone to its number
   pad (123) when asked for a digit.
3. Decode and train:
   ```
   python scripts/decode_iphone_enrollment.py path/to/iphone_session_XXXX.ksiph iphone
   EXPORT=1 python scripts/train.py iphone iphone 0 150 random
   python scripts/build_report.py
   ```
   Logs the found-vs-target count per key as it goes, so a quiet room or a
   key that didn't register taps shows up immediately rather than silently.

Verified end to end (recorder → blind isolation → npz) with a synthetic
session before asking for the real 10–15 minute recording: embedded real
audio for 3 keys at 5 taps each into a test recording, the isolator found
exactly 5/5/5 with zero false positives on the untouched keys.

## Alternative: from the original phone/Zoom recordings

```
EXPORT=1 python scripts/train.py zoom zoom 0 150 random
# or: EXPORT=1 python scripts/train.py phone phone 0 150 random
```

This needs `data/zoom.npz` (or `data/phone.npz`) — the raw isolated strokes
from `scripts/isolate.py` — which are gitignored and currently only exist
locally. `EXPORT=1` only does anything when the train and test set are the
same device (an export trained on one device and run live on another will
sit near chance — see "Tuned to one device" on the page).

It writes:
- `model/<set>/model.onnx` — the trained weights
- `model/<set>/melfb.json` — the exact mel filterbank used in training, so
  the browser's feature extraction (`scripts/mel.js`) matches exactly rather
  than approximately
- `model/<set>/norm.json` — normalization stats (mu, sd), the key order, and
  the training sample rate (the browser resamples live mic audio to this
  before extracting features)

Which device to deploy: `zoom` is the closer match to a live browser mic
capture (laptop's own mic, same as a call); `phone` needs a phone physically
positioned next to the keyboard, which is harder to reproduce live. Either
works technically — pick whichever setup you'll actually have in the room.

After exporting, run `python scripts/build_report.py` again so `index.html`
and `docs/index.html` pick up the new `model/` directory (copied into both,
same as `assets/`).
