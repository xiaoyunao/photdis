#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 01:30:00
#SBATCH --mem=8G
#SBATCH -o logs/cluster_rgb_ngc4147_score_%j.out
#SBATCH -e logs/cluster_rgb_ngc4147_score_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps:$PWD/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/cluster_rgb_ngc4147_score.py
