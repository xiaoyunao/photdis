#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J photdis_v3_0_sed_bench
#SBATCH -o logs/photdis_v3_0_sed_bench_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/v3_0/.sed.lock
flock -n 9
python -u scripts/v3_0_sed.py benchmark
