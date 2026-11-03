#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J v12_finish
#SBATCH -o logs/baseline_v1_2_finish_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${1:-}" != "--report-only" ]]; then
 python scripts/baseline_v1_2_parallax.py
 python scripts/baseline_v1_2_validate.py
fi
python scripts/baseline_v1_2_report.py
