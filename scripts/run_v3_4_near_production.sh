#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=22G
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_4_near_production
#SBATCH -o logs/photdis_v3_4_near_production_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_4_near_production.py "${1:-${SLURM_ARRAY_TASK_ID}}"
