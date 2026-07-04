#!/bin/bash -l
#SBATCH --job-name=nicaff_gdr
#SBATCH --output=nicaff_gdr.o%j
#SBATCH --error=nicaff_gdr.o%j
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
WRAP=/home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs/select_nic.sh
ARGS="--min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 30 --warmup 5"

echo "########## NIC-affinity + GPUDirect (timed, 30 exec) ##########"
mkdir -p $ROOT/gdr_nicaff/csv $ROOT/gdr_nicaff/tts; cd $ROOT/gdr_nicaff
mpirun -np 8 $WRAP $BIN $ARGS 2>&1 | tail -3

echo "########## transport check (proto info, NIC-affinity + GDR) ##########"
mpirun -np 8 -x UCX_PROTO_INFO=y $WRAP $BIN --min-level 8 --max-level 8 \
       --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 1 --warmup 0 2>&1 | \
  grep -iE 'from (host|cuda) memory|gdr_copy|cuda_copy|zcopy|rc_mlx5/mlx5_[0-9]:1' | grep -ivE 'Transport:' | sort -u | head -20
echo "########## END ##########"
