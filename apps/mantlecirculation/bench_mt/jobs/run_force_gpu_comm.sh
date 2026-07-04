#!/bin/bash -l
#SBATCH --job-name=force_gpucomm
#SBATCH --output=force_gpucomm.o%j
#SBATCH --error=force_gpucomm.o%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:20:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export OMP_NUM_THREADS=1
module load openmpi/5.0.5-nvhpc24.11-cuda
BIN=/home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators
ROOT=/hnvme/workspace/iwia054h-mantle/bench_outputs
WRAP=/home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs/select_nic.sh
A="--min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 30 --warmup 5"

echo "###### 0) WHY is GPUDirect declined? (UCX debug, GDR reasons) ######"
mkdir -p $ROOT/force_dbg/csv $ROOT/force_dbg/tts; cd $ROOT/force_dbg
mpirun -np 8 -x UCX_IB_GPU_DIRECT_RDMA=yes -x UCX_LOG_LEVEL=debug $WRAP $BIN \
  --min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 1 --warmup 0 2>&1 | \
  grep -iE 'gpu.?direct|gdr|peer|rndv.*(cuda|scheme)|cuda.*reg|md .*cuda|registration|nv_peer|relaxed' | sort -u | head -25

echo "###### A) get_zcopy (force RDMA zero-copy on GPU mem) ######"
mkdir -p $ROOT/force_getzcopy/csv $ROOT/force_getzcopy/tts; cd $ROOT/force_getzcopy
mpirun -np 8 -x UCX_IB_GPU_DIRECT_RDMA=yes -x UCX_RNDV_SCHEME=get_zcopy $WRAP $BIN $A 2>&1 | tail -2

echo "###### B) RNDV_FRAG_MEM_TYPE=cuda (keep staging frags on GPU) ######"
mkdir -p $ROOT/force_fragcuda/csv $ROOT/force_fragcuda/tts; cd $ROOT/force_fragcuda
mpirun -np 8 -x UCX_IB_GPU_DIRECT_RDMA=yes -x UCX_RNDV_FRAG_MEM_TYPE=cuda $WRAP $BIN $A 2>&1 | tail -2

echo "###### A') transport check for get_zcopy ######"
cd $ROOT/force_getzcopy
mpirun -np 8 -x UCX_IB_GPU_DIRECT_RDMA=yes -x UCX_RNDV_SCHEME=get_zcopy -x UCX_PROTO_INFO=y $WRAP $BIN \
  --min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 --executions 1 --warmup 0 2>&1 | \
  grep -iE 'from (host|cuda) memory|gdr|cuda_copy|zcopy.*cuda|rc_mlx5.*cuda|get_zcopy' | grep -ivE 'Transport:' | sort -u | head -15
echo "###### END ######"
