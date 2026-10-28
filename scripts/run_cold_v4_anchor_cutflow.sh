#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 00:20:00
#SBATCH --mem=24G
#SBATCH -o logs/cold_v4_anchor_cutflow_%j.out
#SBATCH -e logs/cold_v4_anchor_cutflow_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/cold_v4_anchor_cutflow.py
