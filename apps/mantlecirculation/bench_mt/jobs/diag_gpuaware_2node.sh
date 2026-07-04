#!/bin/bash -l
#SBATCH --job-name=diag_gpuaware
#SBATCH --output=diag_gpuaware.o%j
#SBATCH --error=diag_gpuaware.o%j
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

UCXINFO=/apps/SPACK/0.23.0/opt/linux-almalinux9-x86_64_v4/gcc-11.4.1/ucx-1.17.0-x7byp5udsjvl74xayihwiwvazaho5lyo/bin/ucx_info
BIN=/home/hpc/iwia/iwia054h/terraneo-build/apps/benchmarks/performance/benchmark_operators

echo "########## A) PER-NODE CAPABILITY (cuda/gdr transports + GPUDirect-RDMA kernel module) ##########"
srun --ntasks-per-node=1 --ntasks=2 bash -c '
  echo "----- $(hostname) -----"
  echo "[cuda/gdr UCX transports on this node:]"
  '"$UCXINFO"' -d 2>/dev/null | grep -iE "Transport:.*(cuda|gdr)" | sort -u
  echo "[GPUDirect-RDMA kernel module (need nvidia_peermem or nv_peer_mem loaded):]"
  lsmod | grep -iE "nvidia_peermem|nv_peer_mem|gdrdrv" || echo "  NONE LOADED -> GPUDirect RDMA unavailable, inter-node GPU comm stages via host"
'

echo
echo "########## B) ACTUAL HALO PATH: benchmark_operators on 2 nodes with UCX_PROTO_INFO ##########"
echo "Look for the rendezvous (rndv) protocol on cuda buffers: rc_mlx5/gdr_copy = GPU-direct; cuda_copy->sysmem->rc_mlx5 = host-staged."
mpirun -np 8 \
  -x UCX_PROTO_INFO=y \
  -x UCX_LOG_LEVEL=info \
  "$BIN" --min-level 8 --max-level 8 --radial-extra-levels 1 --lat-sdr 1 --rad-sdr 1 \
         --executions 3 --warmup 2 \
  2>&1 | grep -iE 'proto|rndv|cuda|gdr|rc_mlx5|dc_mlx5|sysmem|tag|rma|zcopy|selected|transport' | head -120
echo "########## END ##########"
