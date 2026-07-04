#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=h100
#SBATCH --job-name=mat_sw
#SBATCH --output=mat_sw.o%j
#SBATCH --error=mat_sw.e%j
#SBATCH --time=0:20:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load openmpi/5.0.5-nvhpc24.11-cuda cmake

cd ~/terraneo-build/apps/benchmarks/performance
BENCH=./benchmark_operators
COMMON="--min-level 8 --max-level 8 --executions 5"

# Grid of (lateral, radial) tile sizes, r_passes=1
LATERAL=(1 2 3 4 8)
RADIAL=(2 4 8 16 32 64)

echo "=== Matrix sweep lat × r (r_passes=1) ==="

for LAT in "${LATERAL[@]}"; do
    for R in "${RADIAL[@]}"; do
        TEAM=$((2 * LAT * LAT * R))
        # Skip configs with team > 1024 (hardware max)
        if [ $TEAM -gt 1024 ]; then
            echo "--- (${LAT},${R},1) team=${TEAM} SKIP (too big) ---"
            continue
        fi
        echo "--- (${LAT},${R},1) team=${TEAM} ---"
        srun ${BENCH} ${COMMON} \
            --lat-tile $LAT --r-tile $R --r-passes 1
    done
done

echo "=== Done ==="
