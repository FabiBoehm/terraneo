#!/bin/bash -l
#SBATCH --job-name=cuda_aware_test
#SBATCH --output=cuda_aware_test.o%j
#SBATCH --error=cuda_aware_test.o%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:10:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda

echo "########## CUDA-aware MPI test: device buffer rank0 -> rank1 across 2 nodes ##########"
mpirun -np 2 -x UCX_PROTO_INFO=y ./cuda_aware_test 2>&1 | \
  grep -iE 'GPU-AWARE|WRONG|sending DEVICE|send returned|cudaMalloc FAILED|rndv|from (host|cuda) memory|cuda_copy|gdr_copy|cuda_ipc|rc_mlx5' | \
  grep -ivE 'Transport:' | head -40
echo "########## END ##########"
