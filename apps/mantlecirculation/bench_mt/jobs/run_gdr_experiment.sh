#!/bin/bash -l
#SBATCH --job-name=gdr_expt
#SBATCH --output=gdr_expt.o%j
#SBATCH --error=gdr_expt.o%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:15:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda
BIN=/home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators
ROOT=/hnvme/workspace/iwia054h-mantle/bench_outputs
ARGS="--min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 30 --warmup 5"

echo "########## A) BASELINE (default UCX) ##########"
mkdir -p $ROOT/gdr_baseline/csv $ROOT/gdr_baseline/tts; cd $ROOT/gdr_baseline
mpirun -np 8 $BIN $ARGS 2>&1 | tail -3

echo "########## B) UCX_IB_GPU_DIRECT_RDMA=yes ##########"
mkdir -p $ROOT/gdr_on/csv $ROOT/gdr_on/tts; cd $ROOT/gdr_on
mpirun -np 8 -x UCX_IB_GPU_DIRECT_RDMA=yes $BIN $ARGS 2>&1 | tail -3

echo "########## C) transport check with GPUDirect forced (proto info) ##########"
cd $ROOT/gdr_on
mpirun -np 8 -x UCX_IB_GPU_DIRECT_RDMA=yes -x UCX_PROTO_INFO=y $BIN --min-level 8 --max-level 8 \
       --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 1 --warmup 0 2>&1 | \
  grep -iE 'rendezvous (cuda|pipeline|zero-copy|get|put)|gdr_copy|cuda_copy' | grep -ivE 'Transport:' | sort -u | head -12
echo "########## END ##########"
