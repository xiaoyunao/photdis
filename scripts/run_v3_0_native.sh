#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J photdis_v3_0_native
#SBATCH -o logs/photdis_v3_0_native_%A_%a.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK_MODE=${1:?mode required}
FIELD_INDEX=${SLURM_ARRAY_TASK_ID:?array index required}
mkdir -p data/v3_0/locks
exec 9>"data/v3_0/locks/native_${TASK_MODE}_${FIELD_INDEX}.lock"
flock -n 9
python -u scripts/v3_0_native.py "$TASK_MODE" "$FIELD_INDEX"
