#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -J photdis_v2_revision
#SBATCH -o logs/photdis_v2_revision_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data/v2_0_revision
exec 9>data/v2_0_revision/.lock
flock -n 9
python -u scripts/v2_0_anomalies.py
python -u scripts/v2_0_revision.py base
python -u scripts/v2_0_revision.py combinations --workers 8
python -u scripts/v2_0_revision.py finalize
