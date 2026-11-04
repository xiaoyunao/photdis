#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH -J baseline_v1_4_prepare
#SBATCH -o logs/baseline_v1_4_prepare_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
flock -n data/.baseline_v1_4_native.lock python -u scripts/baseline_v1_4_native.py
flock -n data/.baseline_v1_4_assemble.lock python -u scripts/baseline_v1_4_assemble.py
