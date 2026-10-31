#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -J giant_information
#SBATCH -o logs/giant_information_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps:$PWD/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-}" in
  extract) python scripts/giant_information_extract.py ;;
  fit) python scripts/giant_information_fit.py ;;
  summarize) python scripts/giant_information_summarize.py ;;
  *) echo 'Usage: sbatch scripts/run_giant_information.sh extract|fit|summarize'; exit 2 ;;
esac
