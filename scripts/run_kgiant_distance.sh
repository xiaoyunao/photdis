#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -J kgiant_distance
#SBATCH -o logs/kgiant_distance_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps:$PWD/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-inventory}" in
 inventory) python scripts/kgiant_distance_inventory.py ;;
 fit) python scripts/kgiant_distance_fit.py ;;
 summary) python scripts/kgiant_distance_summary.py ;;
 *) exit 2 ;;
esac
