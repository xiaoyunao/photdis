#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --mem=64G
#SBATCH --array=0-3%2
#SBATCH -o logs/cold_v4_cross_sign_field_%A_%a.out
#SBATCH -e logs/cold_v4_cross_sign_field_%A_%a.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "$SLURM_ARRAY_TASK_ID" in
  0) config=configs/field_cross_ra160_170.json ;;
  1) config=configs/field_cross_ra180_190.json ;;
  2) config=configs/field_cross_ra340_350.json ;;
  3) config=configs/field_cross_ra000_010.json ;;
  *) echo "Unknown array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
python scripts/pilot_field.py --config "$config"
