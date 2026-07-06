#!/bin/bash -l
#SBATCH --job-name=grid_pair
#SBATCH --output=/hnvme/workspace/iwia054h-mantle/slurm_logs/grid_pair.%j.out
#SBATCH --error=/hnvme/workspace/iwia054h-mantle/slurm_logs/grid_pair.%j.out
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:15:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export VISCLOCAL_OUT=$HOME/visclocal_out
echo "### grid-pair render on $(hostname)"
~/.conda/envs/paraview/bin/pvbatch ~/pv_render_grid_pair.py 2>&1 | grep -v -iE 'warn|vtk|deprecat'
echo "### render exit=${PIPESTATUS[0]}"
ls -l "$VISCLOCAL_OUT"/pv/grid_uniform.png "$VISCLOCAL_OUT"/pv/grid_adaptive.png
