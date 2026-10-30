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
