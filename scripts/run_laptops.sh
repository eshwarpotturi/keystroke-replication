#!/bin/bash
# Cross-laptop test: authors' MacBook Pro 16 (M1 Pro) vs a newer MacBook Pro (M3), same 22 kHz sample rate. Starts after the main queue.
cd /home/claude/keystroke-replication
while pgrep -f run_final.sh >/dev/null; do sleep 5; done
run(){ echo "$(date +%FT%T) START $*"; python3 -I scripts/train.py $1 $2 $3 $5 $4 > logs/laptops_$1_$2_$4_$3.log 2>&1; grep RESULT logs/laptops_$1_$2_$4_$3.log; }
run rpractical rmac 0 random 80
run rpractical rpractical 0 random 80
run rmac rpractical 0 random 150
run rmac rmac 0 random 150
echo "$(date +%FT%T) ALL DONE laptops"
