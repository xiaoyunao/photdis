#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J baseline_v1_depthprep
#SBATCH -o logs/baseline_v1_depthprep_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/baseline_v1_depth_prepare.py
