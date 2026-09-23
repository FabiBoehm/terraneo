#!/bin/bash
# Regenerate the LUMI-G standard-mode strong-scaling sweep from the ORIGINAL
# tree-era campaign (~/terraneo/apps/mantlecirculation/bench_mt/jobs on LUMI),
# which is the campaign behind the published numbers.
#
# The decisive difference from the earlier lumi_reproduction scripts: the
# original sets the lateral and radial subdomain levels INDEPENDENTLY
# (--lat-sdr / --rad-sdr), and they are usually unequal, e.g. (0,1) at 4 GCDs,
# (1,0) at 8, (2,0) at 32, (3,1) at 256. Passing a single
# --refinement-level-subdomains N instead applies N to both axes and changes the
# decomposition, which cost +56 % per step in the median and up to +317 %.
# Every point whose decomposition matched reproduced the published value to
# within -4 % / +8 %; every mismatched point was slow.
#
# Everything else follows the original: 10 timesteps, output-frequency 9 (one
# timer_tree_9.json), Stokes 10 FGMRES with restart 10, energy 50, tolerances
# pinned to 0, two pre/post smoothing steps, same environment and GPU binding.
# Only the config differs: ../config_scal_A3.toml, because the original's
# config_fscmb_nsurf_lvl6_10steps.toml no longer parses on the current app.
#
# Overrides: TERRANG_BIN, TERRANG_CFG, TERRANG_OUT.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P="$HERE/points_std_treeera.txt"
OUTROOT=${TERRANG_OUTROOT:-/scratch/project_465002367/bohmfabi/scal_a3_lumi_std}
n=0
while read -r MT G MODE NODES TPN MN MX LAT RAD RX TMIN; do
  [ -z "$MT" ] && continue
  case "$TPN" in
    1) BIND="map_cpu:49" ;;
    2) BIND="map_cpu:49,57" ;;
    4) BIND="map_cpu:49,57,17,25" ;;
    *) BIND="map_cpu:49,57,17,25,1,9,33,41" ;;
  esac
  NAME=MT${MT}_g${G}_std
  cat > "$HERE/std_${NAME}.sh" <<EOF
#!/bin/bash -l
#SBATCH --job-name=ls_${NAME}
#SBATCH --output=${OUTROOT}/logs/${NAME}.o%j
#SBATCH --error=${OUTROOT}/logs/${NAME}.e%j
#SBATCH --partition=standard-g
#SBATCH --account=project_465002367
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=${TPN}
#SBATCH --gpus-per-node=8
#SBATCH --time=00:${TMIN}:00

echo "Cell: ${NAME}  mesh=[${MN}..${MX}]  lat_sdr=${LAT} rad_sdr=${RAD}  steps=10  fgmres=10  ev=50  n_gcds=${G}  nodes=${NODES}x${TPN}"

export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_NO_ASYNC_COPY=1
export OMP_NUM_THREADS=1
export FI_MR_CACHE_MAX_COUNT=1048576
export FI_CXI_RX_MATCH_MODE=software
ulimit -c 0

HERE="\${SLURM_SUBMIT_DIR:-\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)}"
BIN="\${TERRANG_BIN:-/users/bohmfabi/terraneo-mergewt-build/apps/mantlecirculation/mantlecirculation}"
CFG="\${TERRANG_CFG:-\$HERE/../config_scal_A3.toml}"
OUT="\${TERRANG_OUT:-${OUTROOT}/${NAME}}"
mkdir -p "\$OUT" ${OUTROOT}/logs

SELECT_GPU=\${SLURM_SUBMIT_DIR}/select_gpu_\${SLURM_JOB_ID}.sh
cat > \${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=\$SLURM_LOCALID
exec "\$@"
INNER
chmod +x \${SELECT_GPU}
cd "\$OUT"

srun --cpu-bind=${BIND} \${SELECT_GPU} "\$BIN" --config "\$CFG" --extended-parameters \\
  --energy-solver ev \\
  --reference-viscosity 2.459983e25 --radius-cmb 3527020 --radius-surface 6418020 \\
  --temperature-surface 0 --temperature-cmb 3500 \\
  --viscosity-min 1e18 --viscosity-max 1e28 \\
  --refinement-level-mesh-min ${MN} --refinement-level-mesh-max ${MX} \\
  --lat-sdr ${LAT} --rad-sdr ${RAD} --radial-extra-levels ${RX} \\
  --max-timesteps 10 --no-xdmf --no-radial-profiles --output-frequency 9 --dt-min 1e-8 \\
  --stokes-krylov-max-iterations 10 --stokes-krylov-restart 10 \\
  --stokes-krylov-relative-tolerance 0 --stokes-krylov-absolute-tolerance 0 \\
  --energy-krylov-max-iterations 50 \\
  --energy-krylov-relative-tolerance 0 --energy-krylov-absolute-tolerance 0 \\
  --stokes-viscous-pc-num-smoothing-steps-prepost 2 \\
  --outdir "\$OUT" --outdir-overwrite

rm -f \${SELECT_GPU}
EOF
  chmod 644 "$HERE/std_${NAME}.sh"; n=$((n+1))
done < "$P"
echo "generated $n standard-mode scripts"
