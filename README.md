# Keystroke replication: can a phone hear what you type?

**Live page:** https://eshwarpotturi.github.io/keystroke-replication/

Replication of "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (Harrison, Toreini, Mehrnezhad, arXiv 2308.01074) on the authors' public recordings ([JBFH-Dev/Keystroke-Datasets](https://github.com/JBFH-Dev/Keystroke-Datasets)).

## Results
Test presses were never used in training. The number of epochs was chosen on a validation split only; the test split was scored once per run.

| Setup | Paper | Ours |
|---|---|---|
| Phone microphone, paper's random split | 95% | 93.0% mean of 3 runs (88.9 to 95.6) |
| Zoom call, paper's random split | 93% | 91.1% mean of 3 runs (88.9 to 94.4) |
| Phone, train on early presses, test on late presses | - | 84.3% mean of 2 runs (83.3 to 85.2) |
| Zoom, train on early presses, test on late presses | - | 85.6% mean of 2 runs (83.3 to 88.0) |
| Train on phone, test on Zoom | - | 4.6% (1 run; chance is 2.8%) |
| Train on Zoom, test on phone | - | 4.4% (1 run) |
| Train on MacBook Pro 16 (M1 Pro), test on MacBook Pro (M3) | - | 4.9% (1 run; 59.2% on the same laptop's own set) |
| Train on MacBook Pro (M3), test on MacBook Pro 16 (M1 Pro) | - | 2.9% (1 run; 67.3% on its own set, 52 test presses) |

## What it says
1. The headline claim holds in the paper's own setup.
2. A single run is noisy: 90 test presses per run, and runs ranged from 88.9% to 95.6%.
3. The paper's split mixes early and late presses; a stricter split costs about 8 points.
4. A model trained on one device does not carry over to another (near chance), and the same holds across two different MacBook Pros in public recordings from a later study (rooms and microphones differ too, so this is not proof of which factor matters).

## Limits
One person, one laptop, one room. The model is a smaller CoAtNet-style network than the paper's, trained for 150 epochs on a CPU (paper: larger, 1,100 epochs). Phone and Zoom recordings use different sample rates (44.1 kHz and 32 kHz), which may add to the transfer gap. The try-it panel replays held-out presses from the authors' recordings; it does not process audio you supply.

## Run it again
```
python3 scripts/isolate.py                      # 25 presses per key (authors' isolator)
python3 scripts/train.py phone phone 0 150 random
python3 scripts/build_try.py && python3 scripts/build_report.py
```
Datasets are cloned from the authors' repository (not included). Logs carry timestamps in `logs/`.
