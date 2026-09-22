#!/bin/bash
cd /hppfs/scratch/0E/di35guv2/bench_mt/jobs
while true; do
  pending=0
  for f in sng_MT*.sh; do
    c=${f#sng_}; c=${c%.sh}
    [ -f ../outputs/${c}_iso_std/timer_trees/timer_tree_9.json ] && continue
    pending=$((pending+1))
    squeue -u di35guv2 -h -o %j | grep -qx "snb_$c" && continue
    sbatch $f 2>&1 | grep -q Submitted && echo "submitted $c"
  done
  [ "$pending" = "0" ] && { echo DRAINDONE; break; }
  sleep 120
done
