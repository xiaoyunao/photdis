#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -J baseline_v1_4_diagnostics
#SBATCH -o logs/baseline_v1_4_diagnostics_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/.baseline_v1_4_diagnostics.lock
flock -n 9
python -u scripts/baseline_v1_4_diagnostics.py
