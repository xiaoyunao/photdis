#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=80G
#SBATCH -t 01:30:00
#SBATCH -J photdis_v3_5_dr11_spatial
#SBATCH -o logs/photdis_v3_5_dr11_spatial_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_5_dr11_spatial.py
