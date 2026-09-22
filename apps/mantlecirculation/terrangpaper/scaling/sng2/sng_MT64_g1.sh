#!/bin/bash -l
#SBATCH --job-name=snb_MT64_g1
#SBATCH --output=/hppfs/scratch/0E/di35guv2/bench_mt/jobs/snb_MT64_g1.o%j
#SBATCH --error=/hppfs/scratch/0E/di35guv2/bench_mt/jobs/snb_MT64_g1.e%j
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00
#SBATCH --account=pn39jo
module load slurm_setup
module sw stack/24.5.0
module load cmake gcc/14.2.0
export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_RDMA=1
export I_MPI_OFFLOAD_FAST_MEMCPY_COLL=1
export PSM3_RDMA=1
export PSM3_GPUDIRECT=0
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
ulimit -c 0
export TMPDIR=/hppfs/scratch/0E/di35guv2/tmp
OUT=/hppfs/scratch/0E/di35guv2/bench_mt/outputs/MT64_g1_iso_std
mkdir -p "$OUT" /hppfs/scratch/0E/di35guv2/bench_mt/jobs
cd "$OUT"
srun --chdir="$OUT" $HOME/terraneo-build/apps/mantlecirculation/mantlecirculation --energy-solver ev --config $HOME/terraneo/apps/mantlecirculation/parameterfiles/config_fscmb_nsurf_lvl6_10steps.toml --refinement-level-mesh-min 2 --refinement-level-mesh-max 6 --refinement-level-subdomains 0 --radial-extra-levels -1 --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 9 --stokes-krylov-max-iterations 10 --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 --energy-krylov-max-iterations 50 --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 --stokes-krylov-restart 10 --stokes-viscous-pc-num-smoothing-steps-prepost 2 --outdir /hppfs/scratch/0E/di35guv2/bench_mt/outputs/MT64_g1_iso_std --outdir-overwrite
