#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 00:30:00
#SBATCH -J photdis_v2
#SBATCH -o logs/photdis_v2_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/.photdis_v2.lock
flock -n 9
case "${1:?collect, render or finalize}" in
 collect) python -u scripts/v2_0_collect.py ;;
 render) python -u scripts/v2_0_extend.py; python -u scripts/v2_0_figures.py ;;
 finalize) python -u scripts/v2_0_finalize.py ;;
 polish) python -u scripts/v2_0_figures.py --polish; python -u scripts/v2_0_finalize.py ;;
 *) exit 2 ;;
esac
