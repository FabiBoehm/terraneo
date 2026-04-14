#!/bin/bash -l
#SBATCH --job-name=full_sw
#SBATCH --output=full_sw.o%j
#SBATCH --error=full_sw.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --account=project_465002889

export MPICH_GPU_SUPPORT_ENABLED=1
export ROCR_VISIBLE_DEVICES=0

cd /pfs/lustrep3/users/bohmfabi/terraneo-build/apps/benchmarks/performance
BENCH=./benchmark_operators

mkdir -p sweep_results

# Format: LAT R RP
CONFIGS=(
    "1 8 1" "1 16 1" "1 32 1" "1 64 1" "1 128 1"
    "2 4 2" "2 8 2" "2 16 1" "2 16 2" "2 32 1" "2 64 1"
    "3 16 1" "3 16 2"
    "4 4 2" "4 4 4" "4 8 1" "4 8 2" "4 16 1" "4 16 2" "4 32 1"
    "8 4 2" "8 8 2"
)

for cfg in "${CONFIGS[@]}"; do
    read LAT R RP <<< "$cfg"
    TAG="${LAT}_${R}_${RP}"
    echo ""
    echo "=========================================="
    echo "=== Config lat=$LAT r=$R rp=$RP (team=$((2*LAT*LAT*R))) ==="
    echo "=========================================="

    # Step 1: Throughput (3 executions)
    echo "--- Timing ---"
    srun --cpu-bind=map_cpu:49 ${BENCH} --min-level 8 --max-level 8 --executions 3 \
        --lat-tile $LAT --r-tile $R --r-passes $RP 2>/dev/null | grep -E "1 \|.*e\+09" | head -2

    # Step 2: rocprof counters
    echo "--- Profiling ---"
    srun --cpu-bind=map_cpu:49 rocprof \
        -i rocprof_full.txt \
        -o sweep_results/cfg_${TAG}.csv \
        --timestamp on \
        ${BENCH} --min-level 8 --max-level 8 --executions 1 \
        --lat-tile $LAT --r-tile $R --r-passes $RP 2>&1 | tail -2
done

echo ""
echo "=== All done ==="
