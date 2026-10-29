#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH --mem=48G
#SBATCH -o logs/cold_v5_xd_compare_%j.out
#SBATCH -e logs/cold_v5_xd_compare_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/cold_v5_xd_compare.py --components "$1"
