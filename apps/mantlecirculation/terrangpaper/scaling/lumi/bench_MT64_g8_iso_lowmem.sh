#!/bin/bash -l
#SBATCH --job-name=bench_MT64_g8_iso_lowmem
#SBATCH --output=bench_MT64_g8_iso_lowmem.o%j
#SBATCH --error=bench_MT64_g8_iso_lowmem.e%j
#SBATCH -D /pfs/lustrep3/users/bohmfabi/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=standard-g
#SBATCH --account=project_465002367
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00

echo "Cell: MT64_g8_iso_lowmem  mesh=[2..6]  rad_level=5  lat_sdr=1  rad_sdr=1  subdomains=80  subdom/GCD=10  steps=10  fgmres=10  ev=50  n_gpus=8  nodes=1x8  partition=standard-g"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
# Raise the libfabric memory-registration cache ceiling. At high rank counts the
# halo exchange opens >10000 active MRs, exhausting the CXI provider default and
# tripping a Cray MPICH internal assertion (cray_ch4_mem_utils.c) + segfault in
# the first energy solve. Lifting the cache count (and the per-region cap) lets
# the >=2048-rank cells register all their halo buffers.
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

# Per-GCD GPU binding wrapper (maps SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=${SLURM_SUBMIT_DIR}/select_gpu_${SLURM_JOB_ID}.sh
cat > ${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${SELECT_GPU}

# mantlecirculation writes its output tree under --outdir; cd there to capture any
# CWD-relative artifacts (timing trees) too.
mkdir -p /pfs/lustrep3/users/bohmfabi/terraneo/apps/mantlecirculation/bench_mt/outputs/MT64_g8_iso_lowmem
cd /pfs/lustrep3/users/bohmfabi/terraneo/apps/mantlecirculation/bench_mt/outputs/MT64_g8_iso_lowmem

srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 ${SELECT_GPU} /pfs/lustrep3/users/bohmfabi/terraneo-build/apps/mantlecirculation/mantlecirculation --config /pfs/lustrep3/users/bohmfabi/terraneo/apps/mantlecirculation/parameterfiles/config_fscmb_nsurf_lvl6_10steps.toml --refinement-level-mesh-min 2 --refinement-level-mesh-max 6 --refinement-level-subdomains 1 --radial-extra-levels -1 --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 11 --stokes-krylov-max-iterations 10 --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 --energy-krylov-max-iterations 50 --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 --stokes-float-krylov-basis --energy-float-krylov-basis --stokes-krylov-restart 5 --energy-krylov-restart 5 --stokes-viscous-pc-num-smoothing-steps-prepost 1 --outdir /pfs/lustrep3/users/bohmfabi/terraneo/apps/mantlecirculation/bench_mt/outputs/MT64_g8_iso_lowmem --outdir-overwrite

rm -f ${SELECT_GPU}
