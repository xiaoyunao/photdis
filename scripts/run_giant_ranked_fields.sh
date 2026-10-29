#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH --mem=64G
#SBATCH --array=0-2%2
#SBATCH -o logs/giant_ranked_field_%A_%a.out
#SBATCH -e logs/giant_ranked_field_%A_%a.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "$SLURM_ARRAY_TASK_ID" in
  0) config=configs/field_giant_ranked_ra200_210_dec_p000_p010.json ;;
  1) config=configs/field_giant_ranked_ra190_200_dec_p000_p010.json ;;
  2) config=configs/field_giant_ranked_ra210_220_dec_p040_p050.json ;;
  *) echo "Unknown array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
python scripts/pilot_field.py --config "$config"
