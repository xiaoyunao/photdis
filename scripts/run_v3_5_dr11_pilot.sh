#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 48
#SBATCH --mem=42G
#SBATCH -t 00:29:00
#SBATCH -J photdis_v3_5_dr11_pilot
#SBATCH -o logs/photdis_v3_5_dr11_pilot_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_5_dr11_pilot.py "$@"
