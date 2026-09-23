#!/bin/bash
# Regenerate the LUMI-G strong-scaling sweep of Paper A against the CURRENT app.
# Same 118 points (59 configs x std/lowmem) as the archived scripts in ../lumi/,
# same environment, srun binding and solver settings; only the config and the
# output-frequency differ: ../config_scal_A3.toml instead of the retired
# config_fscmb_nsurf_lvl6_10steps.toml, and --output-frequency 9 so that the ten
# timesteps (0..9) write exactly one timer_trees/timer_tree_9.json. The archived
# scripts used frequency 11 and therefore never wrote a usable tree.
#
# Overrides for a checkout elsewhere: TERRANG_BIN, TERRANG_CFG, TERRANG_OUT.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POINTS="$HERE/points.txt"
[ -f "$POINTS" ] || { echo "missing $POINTS"; exit 1; }
n=0
while read -r MT G MODE NODES TPN MN MX SD RX TMIN; do
  [ -z "$MT" ] && continue
  NAME=MT${MT}_g${G}_${MODE}
  if [ "$MODE" = lowmem ]; then
    SOLV='--stokes-float-krylov-basis --energy-float-krylov-basis --stokes-krylov-restart 5 --energy-krylov-restart 5 --stokes-viscous-pc-num-smoothing-steps-prepost 1'
  else
    SOLV='--stokes-krylov-restart 10 --stokes-viscous-pc-num-smoothing-steps-prepost 2'
  fi
  cat > "$HERE/run_${NAME}.sh" <<EOF
#!/bin/bash -l
#SBATCH --job-name=lr_${NAME}
#SBATCH --output=/scratch/project_465002367/bohmfabi/scal_a3_lumi/logs/${NAME}.o%j
#SBATCH --error=/scratch/project_465002367/bohmfabi/scal_a3_lumi/logs/${NAME}.e%j
#SBATCH --partition=standard-g
#SBATCH --account=project_465002367
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${TPN}
#SBATCH --gpus-per-node=8
#SBATCH --time=00:${TMIN}:00

echo "Cell: ${NAME}  mesh=[${MN}..${MX}]  subdomains-level=${SD}  steps=10  fgmres=10  ev=50  n_gcds=${G}  nodes=${NODES}x${TPN}"

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
HERE="\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)}"
BIN="\${TERRANG_BIN:-/users/bohmfabi/terraneo-mergewt-build/apps/mantlecirculation/mantlecirculation}"
CFG="\${TERRANG_CFG:-\$HERE/../config_scal_A3.toml}"
OUT="\${TERRANG_OUT:-/scratch/project_465002367/bohmfabi/scal_a3_lumi/${NAME}}"
mkdir -p "\$OUT" /scratch/project_465002367/bohmfabi/scal_a3_lumi/logs

# Per-GCD GPU binding wrapper (SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=\${SLURM_SUBMIT_DIR}/select_gpu_\${SLURM_JOB_ID}.sh
cat > \${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec "\$@"
INNER
chmod +x \${SELECT_GPU}
cd "\$OUT"

srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 \${SELECT_GPU} "\$BIN" --config "\$CFG" --extended-parameters \\
  --energy-solver ev \\
  --reference-viscosity 2.459983e25 --radius-cmb 3527020 --radius-surface 6418020 \\
  --temperature-surface 0 --temperature-cmb 3500 \\
  --viscosity-min 1e18 --viscosity-max 1e28 \\
  --refinement-level-mesh-min ${MN} --refinement-level-mesh-max ${MX} \\
  --refinement-level-subdomains ${SD} --radial-extra-levels ${RX} \\
  --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 9 --dt-min 1e-8 \\
  --stokes-krylov-max-iterations 10 \\
  --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 \\
  --energy-krylov-max-iterations 50 \\
  --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 \\
  ${SOLV} \\
  --outdir "\$OUT" --outdir-overwrite

rm -f \${SELECT_GPU}
EOF
  chmod 644 "$HERE/run_${NAME}.sh"; n=$((n+1))
done < "$POINTS"
echo "generated $n scripts in $HERE"
