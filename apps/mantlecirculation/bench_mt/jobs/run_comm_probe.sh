#!/bin/bash -l
#SBATCH --job-name=comm_probe
#SBATCH --output=comm_probe.o%j
#SBATCH --error=comm_probe.o%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:10:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda
BIN=/home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators

echo "########## COMM BUFFER MEMORY-SPACE PROBE (2 nodes / 8 GPU, g8 cell) ##########"
mpirun -np 8 -x UCX_PROTO_INFO=y "$BIN" --min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 \
       --executions 2 --warmup 1 2>&1 | grep -iE 'comm-probe|from (host|cuda) memory|cuda_copy|gdr_copy|rc_mlx5|knem' | grep -ivE 'Transport:'
echo "########## END ##########"
