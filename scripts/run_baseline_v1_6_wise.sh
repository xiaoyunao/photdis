#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH -J photdis_weak_wise
#SBATCH -o logs/photdis_weak_wise_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/.photdis_weak_wise.lock
flock -n 9
case "${1:?raw, fit or report}" in
  raw) python -u scripts/baseline_v1_6_wise_raw.py ;;
  fit) python -u scripts/baseline_v1_6_wise.py ;;
  report) python -u scripts/baseline_v1_6_wise_report.py ;;
  *) exit 2 ;;
esac
