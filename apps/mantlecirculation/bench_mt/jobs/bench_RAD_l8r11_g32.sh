#!/bin/bash -l
#SBATCH --job-name=bench_RAD_l8r11_g32
#SBATCH --output=bench_RAD_l8r11_g32.o%j
#SBATCH --error=bench_RAD_l8r11_g32.e%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:30:00
#SBATCH --export=NONE

echo "Cell: RAD_l8r11_g32  lat_level=8  rad_level=11  lat_sdr=1  rad_sdr=3  subdomains=320  n_gpus=32  nodes=8x4  partition=h100"

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda

# benchmark_operators writes csv/ and tts/ relative to CWD.
mkdir -p /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r11_g32/csv /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r11_g32/tts
cd /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r11_g32

mpirun -np 32 /home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators --min-level 8 --max-level 8 --radial-extra-levels 3 --lat-sdr 1 --rad-sdr 3 --executions 30 --warmup 5
