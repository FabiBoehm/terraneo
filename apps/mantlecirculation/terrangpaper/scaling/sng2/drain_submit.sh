#!/bin/bash
cd /hppfs/scratch/0E/di35guv2/bench_mt/jobs
for c in MT64_g16 MT64_g1 MT64_g2 MT64_g32 MT64_g4 MT64_g8; do
  while true; do
    out=$(sbatch sng_$c.sh 2>&1)
    echo "$out" | grep -q Submitted && { echo "$c: $out"; break; }
    sleep 120
  done
done
echo DRAINDONE
