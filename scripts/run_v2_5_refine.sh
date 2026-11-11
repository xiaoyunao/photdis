#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J photdis_v2_5_refine
#SBATCH -o logs/photdis_v2_5_refine_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
exec 9>data/v2_5/.run.lock
flock -n 9
python -u scripts/v2_5_refine.py
python -u scripts/v2_5_validate.py
