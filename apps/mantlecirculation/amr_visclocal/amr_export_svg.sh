#!/bin/bash -l
#SBATCH --job-name=export_svg
#SBATCH --output=/hnvme/workspace/iwia054h-mantle/slurm_logs/export_svg.%j.out
#SBATCH --error=/hnvme/workspace/iwia054h-mantle/slurm_logs/export_svg.%j.out
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:25:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
export VISCLOCAL_OUT=$HOME/visclocal_out
echo "### GL2PS vector SVG export on $(hostname)"
~/.conda/envs/paraview/bin/pvbatch ~/pv_export_svg.py 2>&1 | grep -v -iE 'warn|vtk|deprecat'
echo "### export exit=${PIPESTATUS[0]}"
ls -l "$VISCLOCAL_OUT"/pv/*.svg 2>/dev/null
