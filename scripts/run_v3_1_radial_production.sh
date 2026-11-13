#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --array=0-1%2
#SBATCH -t 04:00:00
#SBATCH -J photdis_v3_1_radialall
#SBATCH -o logs/photdis_v3_1_radialall_%A_%a.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_1_radial_production.py --shard "$SLURM_ARRAY_TASK_ID"
