#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J photdis_v2_2_validate
#SBATCH -o logs/photdis_v2_2_validate_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/v2_2/.validate.lock
flock -n 9
python -u scripts/v2_2_noise_summary.py
python -u scripts/v2_2_extremes.py
python -u scripts/v2_2_selected_gaia.py
python -u scripts/v2_2_validate.py
