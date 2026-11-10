#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J photdis_v2_2_application
#SBATCH -o logs/photdis_v2_2_application_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data/v2_2
exec 9>data/v2_2/.application.lock
flock -n 9
python -u scripts/v2_2_application.py
