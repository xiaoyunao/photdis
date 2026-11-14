#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 00:15:00
#SBATCH -J photdis_v3_2_tilt_pilot
#SBATCH -o logs/photdis_v3_2_tilt_pilot_%j.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_2_tilt_pilot.py "${1:?Output directory required}"
