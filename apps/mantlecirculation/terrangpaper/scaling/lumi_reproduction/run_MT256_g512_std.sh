#!/bin/bash -l
#SBATCH --job-name=lr_MT256_g512_std
#SBATCH --output=/scratch/project_465002367/bohmfabi/scal_a3_lumi/logs/MT256_g512_std.o%j
#SBATCH --error=/scratch/project_465002367/bohmfabi/scal_a3_lumi/logs/MT256_g512_std.e%j
#SBATCH --partition=standard-g
#SBATCH --account=project_465002367
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:15:00

echo "Cell: MT256_g512_std  mesh=[4..8]  subdomains-level=3  steps=10  fgmres=10  ev=50  n_gcds=512  nodes=64x8"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
# libfabric MR cache ceiling: >=2048-rank cells exhaust the CXI default.
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

# Paths. Override to run from another checkout:
#   TERRANG_BIN  the mantlecirculation binary
#   TERRANG_CFG  config_scal_A3.toml (defaults to the copy next to this tree)
#   TERRANG_OUT  output directory for this point
HERE="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BIN="${TERRANG_BIN:-/users/bohmfabi/terraneo-mergewt-build/apps/mantlecirculation/mantlecirculation}"
CFG="${TERRANG_CFG:-$HERE/../config_scal_A3.toml}"
OUT="${TERRANG_OUT:-/scratch/project_465002367/bohmfabi/scal_a3_lumi/MT256_g512_std}"
mkdir -p "$OUT" /scratch/project_465002367/bohmfabi/scal_a3_lumi/logs

# Per-GCD GPU binding wrapper (SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=${SLURM_SUBMIT_DIR}/select_gpu_${SLURM_JOB_ID}.sh
cat > ${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${SELECT_GPU}
cd "$OUT"

srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 ${SELECT_GPU} "$BIN" --config "$CFG" --extended-parameters \
  --energy-solver ev \
  --reference-viscosity 2.459983e25 --radius-cmb 3527020 --radius-surface 6418020 \
  --temperature-surface 0 --temperature-cmb 3500 \
  --viscosity-min 1e18 --viscosity-max 1e28 \
  --refinement-level-mesh-min 4 --refinement-level-mesh-max 8 \
  --refinement-level-subdomains 3 --radial-extra-levels -1 \
  --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 9 --dt-min 1e-8 \
  --stokes-krylov-max-iterations 10 \
  --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 \
  --energy-krylov-max-iterations 50 \
  --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 \
  --stokes-krylov-restart 10 --stokes-viscous-pc-num-smoothing-steps-prepost 2 \
  --outdir "$OUT" --outdir-overwrite

rm -f ${SELECT_GPU}
