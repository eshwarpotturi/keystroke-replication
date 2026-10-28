# Keystroke replication

Replication of "A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards" (arXiv 2308.01074) on the authors' public recordings (JBFH-Dev/Keystroke-Datasets).

**Deployed page:** https://eshwarpotturi.github.io/keystroke-replication/

## Results (test strokes held out; epochs chosen on validation only)
| Setup | Paper | Ours |
|---|---|---|
| Phone mic, random split, 150 epochs | 95% | 93.0% mean of 3 runs (88.9 to 95.6) |
| Zoom, random split, 150 epochs | 93% | 92.2% mean of 2 runs (90.0 to 94.4) |
| Phone, random split, 90 epochs (matched baseline) | - | 87.8% (1 run) |
| Phone, early presses train / late presses test, 90 epochs | - | 73.1% (1 run) |

## Not done
Zoom early-vs-late, phone-to-Zoom transfer, more seeds on the stricter test. One person, one laptop, one room.

## Run
```
python3 scripts/isolate.py      # 25 presses per key, authors' isolator
python3 scripts/train.py phone phone 0 150 random
python3 scripts/build_report.py
```
Model is a smaller CoAtNet-style network than the paper's; CPU only. Logs carry timestamps in `logs/`.
