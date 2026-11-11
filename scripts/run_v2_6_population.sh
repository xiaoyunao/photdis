#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J photdis_v2_6_population
#SBATCH -o logs/photdis_v2_6_population_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p data/v2_6
exec 9>data/v2_6/.population.lock
flock -n 9
python -u scripts/v2_6_population.py
