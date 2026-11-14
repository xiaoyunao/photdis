#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_2_tilt_fit
#SBATCH -o logs/photdis_v3_2_tilt_fit_%A_%a.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${SLURM_ARRAY_TASK_ID:?Arm index required}" in
  0) arm=(broad zero) ;;
  1) arm=(broad free) ;;
  2) arm=(candidate zero) ;;
  3) arm=(candidate free) ;;
  *) exit 2 ;;
esac
python -u scripts/v3_2_tilt_fit.py "${arm[@]}" "$@"
