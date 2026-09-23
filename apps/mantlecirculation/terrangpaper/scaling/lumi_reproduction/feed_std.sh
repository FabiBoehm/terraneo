#!/bin/bash
# Drip-feed the LUMI-G standard-mode sweep in ascending node count. Idempotent:
# skips points that already produced timer_trees/timer_tree_9.json or are queued,
# and backs off on any sbatch rejection.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${TERRANG_OUTROOT:-/scratch/project_465002367/bohmfabi/scal_a3_lumi_std}
MAX_NODES=${MAX_NODES:-1024}
LOG=$ROOT/feed_std.log
mkdir -p "$ROOT/logs"
echo "feeder start $(date +%F' '%T) max_nodes=$MAX_NODES" >> "$LOG"
while true; do
  pend=0
  for f in $(for g in "$HERE"/std_MT*.sh; do echo "$(grep -oP '(?<=--nodes=)\d+' "$g") $g"; done | sort -n | awk '{print $2}'); do
    nm=$(basename "$f"); nm=${nm#std_}; nm=${nm%.sh}
    nodes=$(grep -oP '(?<=--nodes=)\d+' "$f")
    [ "$nodes" -gt "$MAX_NODES" ] && continue
    [ -f "$ROOT/$nm/timer_trees/timer_tree_9.json" ] && continue
    pend=$((pend+1))
    squeue -u "$USER" -h -n "ls_$nm" -o "%T" | grep -q . && continue
    out=$(cd "$HERE" && sbatch "$f" 2>&1)
    if echo "$out" | grep -q "Submitted batch job"; then
      echo "$(date +%T) $nm ($nodes n): ${out##* }" >> "$LOG"
    else
      echo "$(date +%T) hold $nm: $(echo "$out" | head -1)" >> "$LOG"; break
    fi
  done
  [ "$pend" -eq 0 ] && { echo "ALL COMPLETE $(date +%F' '%T)" >> "$LOG"; break; }
  sleep 120
done
