#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -J photdis_v3_2_structure_odds
#SBATCH -o logs/photdis_v3_2_structure_odds_%j.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u scripts/v3_2_structure_odds_audit.py
