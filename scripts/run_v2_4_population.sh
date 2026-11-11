#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 02:00:00
#SBATCH -J photdis_v2_4_population
#SBATCH -o logs/photdis_v2_4_population_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data/v2_4
exec 9>data/v2_4/.population.lock
flock -n 9
python -u scripts/v2_4_population.py application
python -u scripts/v2_4_population.py reserve
python -u scripts/v2_4_recovery.py
