#!/bin/bash
# Drip-feed the LUMI reproduction sweep. Submits in ascending node count, skips
# points that already have timer_trees/timer_tree_9.json or are in the queue,
# stops at MAX_NODES (default 1024, which is standard-g's MaxNodes: the four
# 2048-node points cannot be submitted to that partition at all and would need
# a reservation), and backs off on any sbatch rejection. Rerun freely; idempotent.
# MODE selects the series: std (default) or lowmem; set MODE=all for both.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${TERRANG_OUTROOT:-/scratch/project_465002367/bohmfabi/scal_a3_lumi}
MAX_NODES=${MAX_NODES:-1024}
MODE=${MODE:-std}
LOG=$ROOT/feed_lumi.log
mkdir -p "$ROOT/logs"
echo "feeder start $(date +%F' '%T) max_nodes=$MAX_NODES mode=$MODE" >> "$LOG"
while true; do
  pend=0; held=0
  case "$MODE" in all) PAT="run_MT*.sh";; *) PAT="run_MT*_${MODE}.sh";; esac
  # ascending node count, so the cheap points fill the queue first
  for f in $(for g in "$HERE"/$PAT; do echo "$(grep -oP '(?<=--nodes=)\d+' "$g") $g"; done | sort -n | awk '{print $2}'); do
    nm=$(basename "$f"); nm=${nm#run_}; nm=${nm%.sh}
    nodes=$(grep -oP '(?<=--nodes=)\d+' "$f")
    [ "$nodes" -gt "$MAX_NODES" ] && continue
    [ -f "$ROOT/$nm/timer_trees/timer_tree_9.json" ] && continue
    pend=$((pend+1))
    squeue -u "$USER" -h -n "lr_$nm" -o "%T" | grep -q . && continue
    out=$(cd "$HERE" && sbatch "$f" 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "$(date +%T) $nm ($nodes n): ${out##* }" >> "$LOG"
    else
      echo "$(date +%T) hold $nm: $(echo "$out" | head -1)" >> "$LOG"; held=1; break
    fi
  done
  [ "$pend" -eq 0 ] && { echo "ALL COMPLETE $(date +%F' '%T)" >> "$LOG"; break; }
  [ "$held" -eq 0 ] && [ "$pend" -gt 0 ] && sleep 120 || sleep 120
done
