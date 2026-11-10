#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J photdis_v2_3_1_correct
#SBATCH -o logs/photdis_v2_3_1_correct_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/v2_3/.inference_correction.lock
flock -n 9
python -u scripts/v2_3_1_migrate.py
for role in development reserved application; do python -u scripts/v2_3_run.py "$role" --candidate-only; done
python -u scripts/v2_3_diagnostics.py
python -u scripts/v2_3_far.py
python -u scripts/v2_3_validate.py
