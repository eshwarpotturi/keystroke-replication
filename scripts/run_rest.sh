#!/bin/bash
# faster queue for the extra checks: 120 epochs, one seed for transfer. Waits for the Zoom seed-2 run already in progress.
cd /home/claude/keystroke-replication
while pgrep -f "train.py zoom zoom 2" >/dev/null; do sleep 5; done
echo "$(date +%FT%T) zoom seed 2 finished"; grep RESULT logs/run_zoom_zoom_random_2.log
run(){ echo "$(date +%FT%T) START $*"; python3 -I scripts/train.py $1 $2 $3 120 $4 > logs/run_$1_$2_$4_$3.log 2>&1; grep RESULT logs/run_$1_$2_$4_$3.log; }
for s in 0 1; do run phone phone $s ordered; run zoom zoom $s ordered; done
run phone zoom 0 random; run zoom phone 0 random
echo "$(date +%FT%T) ALL DONE"
