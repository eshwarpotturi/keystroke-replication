# Keystroke Listener

A compact model that learns what each key sounds like from 25 recorded presses, then names presses it has never heard, through a phone microphone or a Zoom call.

Live demo: https://eshwarpotturi.github.io/keystroke-replication/

## What it does well

| Test (36 keys, chance 2.8%) | Keys named correctly |
|---|---|
| Phone microphone, mixed presses | 93.0% (published: 95%) |
| Zoom call, mixed presses | 91.1% (published: 93%) |
| Phone / Zoom, trained on first 20 presses, tested on last 3 | 84.3% / 85.6% |
| When the model is 80%+ sure | 98.0% right, on 63% of presses |
| When the model is 90%+ sure | 99.2% right, on 33% of presses |
| Right key within its top three guesses | 97.5% of presses |

- Learns a device from about 900 presses and trains in minutes on a CPU.
- Knows when it knows: a confidence gate keeps the sure keys and passes the rest on.
- Every run is saved and shown on the page, with a spectrogram and audio for each test press.

## What you can use it for

- Test how much your own keyboard, laptop and microphone give away by sound.
- Compare keyboards, mounts or microphones with the same test.
- Add a confidence gate or top-three shortlist to a downstream step.

These are possible uses; none has been tried with a client.

## Where it fits

Each model is tuned to the device and room it learned from, so train it on the device you want to measure. On a different device it scores about 4.5% (phone to Zoom, Zoom to phone) and about 4% across two different MacBook Pros (single runs; sampling rates and rooms also differ). Data is one person, one laptop, one room.

## Run it

```
python scripts/isolate.py        # find each key press in the recordings
python scripts/train.py phone phone 0 150 random
python scripts/build_report.py   # writes index.html
```

## Live tab

The page has a third tab, "Live (your mic)": type a passage, then type a
hidden password, and it guesses the password back from your own microphone,
live, entirely in the browser (nothing is sent anywhere). It only works on
the device and mic a model was trained on — see `model/README.md` for how to
export one; until an export exists, the tab says so rather than pretending
to work. `scripts/mel.js` is a JS port of the feature pipeline in
`train.py`'s `mel_fn`, checked against it numerically in
`scripts/test_mel_js.py`.

## Credit

Method and data: "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (arXiv 2308.01074) and the JBFH-Dev/Keystroke-Datasets recordings. R-ASCA laptop data from przybytniowskaj/KeystrokeDetection. This is an independent re-implementation with a smaller model.
