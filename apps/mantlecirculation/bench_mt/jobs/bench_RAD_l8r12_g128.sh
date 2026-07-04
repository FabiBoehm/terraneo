#!/bin/bash -l
#SBATCH --job-name=bench_RAD_l8r12_g128
#SBATCH --output=bench_RAD_l8r12_g128.o%j
#SBATCH --error=bench_RAD_l8r12_g128.e%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:30:00
#SBATCH --export=NONE

echo "Cell: RAD_l8r12_g128  lat_level=8  rad_level=12  lat_sdr=1  rad_sdr=5  subdomains=1280  n_gpus=128  nodes=32x4  partition=h100"

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda

# benchmark_operators writes csv/ and tts/ relative to CWD.
mkdir -p /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r12_g128/csv /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r12_g128/tts
cd /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r12_g128

mpirun -np 128 /home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators --min-level 8 --max-level 8 --radial-extra-levels 4 --lat-sdr 1 --rad-sdr 5 --executions 5 --warmup 5
