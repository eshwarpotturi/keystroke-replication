#!/bin/bash
# queue of all remaining runs; each run writes logs/res_*.json
cd /home/claude/keystroke-replication
run(){ f="logs/res_$1_$2_$4_$3.json"; [ -f "$f" ] && return; echo "$(date +%FT%T) START $*"; python3 -I scripts/train.py $1 $2 $3 150 $4 > logs/run_$1_$2_$4_$3.log 2>&1; grep RESULT logs/run_$1_$2_$4_$3.log; }
for s in 1 2; do run phone phone $s random; done
for s in 0 1 2; do run zoom zoom $s random; done
for s in 0 1; do run phone phone $s ordered; run zoom zoom $s ordered; done
for s in 0 1; do run phone zoom $s random; run zoom phone $s random; done
echo "$(date +%FT%T) ALL DONE"
