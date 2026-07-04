#!/bin/bash -l
#SBATCH --job-name=bench_RAD_l8r7_g8
#SBATCH --output=bench_RAD_l8r7_g8.o%j
#SBATCH --error=bench_RAD_l8r7_g8.e%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:15:00
#SBATCH --export=NONE

echo "Cell: RAD_l8r7_g8  lat_level=8  rad_level=7  lat_sdr=1  rad_sdr=1  subdomains=80  n_gpus=8  nodes=2x4  partition=h100"

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda

# benchmark_operators writes csv/ and tts/ relative to CWD.
mkdir -p /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r7_g8/csv /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r7_g8/tts
cd /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r7_g8

mpirun -np 8 /home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators --min-level 8 --max-level 8 --radial-extra-levels -1 --lat-sdr 1 --rad-sdr 1 --executions 30 --warmup 5
