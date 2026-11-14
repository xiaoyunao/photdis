#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_2_reference_final
#SBATCH -o logs/photdis_v3_2_reference_final_%j.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
decision=${1:?Expected evaluated mechanism receipt}
forecast=data/v3_2/reference_forecast/complete.json
evaluator=scripts/v3_2_reference_score.py
python -u scripts/v3_2_reference_freeze.py "$decision" "$forecast" "$evaluator"
python -u scripts/v3_2_reference_evidence.py
python -u "$evaluator"
