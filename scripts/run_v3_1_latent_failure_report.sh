#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J photdis_v3_1_latentfailure
#SBATCH -o logs/photdis_v3_1_latentfailure_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_1_latent_failure_report.py
