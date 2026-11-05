#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J photdis_r0_report
#SBATCH -o logs/photdis_r0_report_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/.photdis_r0_report.lock
flock -n 9
python -u scripts/baseline_v1_5_r0_report.py
