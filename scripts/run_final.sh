#!/bin/bash
# final queue: everything at 150 epochs, 2 threads, run one after another. Each run writes logs/res_*.json and data/pred_*.npz
cd /home/claude/keystroke-replication
run(){ echo "$(date +%FT%T) START $*"; python3 -I scripts/train.py $1 $2 $3 150 $4 > logs/final_$1_$2_$4_$3.log 2>&1; grep RESULT logs/final_$1_$2_$4_$3.log; }
run phone phone 0 random      # re-run so test predictions are saved for the try-it panel
run zoom zoom 0 random
run phone phone 0 ordered; run zoom zoom 0 ordered
run phone zoom 0 random; run zoom phone 0 random
run zoom zoom 2 random
run phone phone 1 ordered; run zoom zoom 1 ordered
echo "$(date +%FT%T) ALL DONE"
