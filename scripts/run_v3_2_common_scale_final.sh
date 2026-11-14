#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -J photdis_v3_2_common_final
#SBATCH -o logs/photdis_v3_2_common_final_%j.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
decision=${1:?Expected evaluated mechanism receipt}
candidate=${2:?Expected candidate common-system forecast receipt}
python -u scripts/v3_2_reference_forecast.py "$decision" "$candidate" --common
python -u scripts/v3_2_common_scale_score.py "$decision"
