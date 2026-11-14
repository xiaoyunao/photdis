#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 12
#SBATCH -t 00:30:00
#SBATCH -J photdis_v3_2_tilt_refine
#SBATCH -o logs/photdis_v3_2_tilt_refine_%A_%a.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_2_tilt_refine.py --shard "${SLURM_ARRAY_TASK_ID:?Shard array required}" "$@"
