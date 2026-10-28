#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --mem=32G
#SBATCH -o logs/cold_v4_cross_sign_score_%j.out
#SBATCH -e logs/cold_v4_cross_sign_score_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/cold_v4_native_transfer_supplement.py \
  --registration configs/cold_v4_native_transfer_cross_sign.json \
  --field-key diagnostic_fields --output cross_sign
