#!/bin/bash -l
#SBATCH --job-name=nvsmi_topo
#SBATCH --output=nvsmi_topo.o%j
#SBATCH --error=nvsmi_topo.o%j
#SBATCH -D /home/hpc/iwia/iwia054h/terraneo/apps/mantlecirculation/bench_mt/jobs
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --time=00:05:00
#SBATCH --export=NONE
echo "===== nvidia-smi (brief) ====="
nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv
echo "===== nvidia-smi topo -m (GPU <-> NIC connectivity) ====="
nvidia-smi topo -m
echo "===== END ====="
