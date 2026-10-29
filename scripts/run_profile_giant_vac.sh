#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:20:00
#SBATCH --mem=4G
#SBATCH -o logs/giant_vac_profile_%j.out
#SBATCH -e logs/giant_vac_profile_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/profile_giant_vac.py
