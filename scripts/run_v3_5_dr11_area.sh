#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=22G
#SBATCH -t 00:24:00
#SBATCH -J photdis_v3_5_dr11_area
#SBATCH -o logs/photdis_v3_5_dr11_area_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_5_dr11_area.py "$@"
