#!/bin/bash -l
#SBATCH --job-name=bench_RAD_l8r16_g2048
#SBATCH --output=bench_RAD_l8r16_g2048.o%j
#SBATCH --error=bench_RAD_l8r16_g2048.e%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=standard-g
#SBATCH --account=project_465002367
#SBATCH --nodes=256
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00

echo "Cell: RAD_l8r16_g2048  lat_level=8  rad_level=16  lat_sdr=1  rad_sdr=9  subdomains=20480  n_gpus=2048  nodes=256x8  partition=standard-g"

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
ulimit -c 0

# Per-GCD GPU binding wrapper (maps SLURM_LOCALID -> ROCR_VISIBLE_DEVICES).
SELECT_GPU=${SLURM_SUBMIT_DIR}/select_gpu_${SLURM_JOB_ID}.sh
cat > ${SELECT_GPU} << 'INNER'
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
INNER
chmod +x ${SELECT_GPU}

# benchmark_operators writes csv/ and tts/ relative to CWD.
mkdir -p /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r16_g2048/csv /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r16_g2048/tts
cd /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/outputs/RAD_l8r16_g2048

srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 ${SELECT_GPU} /pfs/lustrep3/users/bohmfabi/terraneo-build/apps/benchmarks/performance/benchmark_operators --min-level 8 --max-level 8 --radial-extra-levels 8 --lat-sdr 1 --rad-sdr 9 --executions 5 --warmup 5

rm -f ${SELECT_GPU}
