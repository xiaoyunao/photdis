#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -J photdis_v2_3_model
#SBATCH -o logs/photdis_v2_3_model_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data/v2_3
exec 9>data/v2_3/.model.lock
flock -n 9
if [ ! -f data/v2_3/models_frozen.json ]; then
  python -u scripts/v2_3_run.py fit
fi
python -u scripts/v2_3_run.py development
python -u scripts/v2_3_run.py reserved
python -u scripts/v2_3_run.py application
