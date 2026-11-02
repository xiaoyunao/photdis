#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J giant_wide
#SBATCH -o logs/giant_wide_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps:$PWD/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:-fit}" in
 fit) python scripts/giant_wide_run.py ;;
 summarize) python scripts/giant_wide_summarize.py ;;
 distances) python scripts/distance_comparison_plot.py ;;
 validate) python scripts/giant_wide_validate.py ;;
 render) python scripts/distance_comparison_plot.py --render-only ;;
 cmd) python scripts/distance_comparison_cmd.py ;;
 finish) python scripts/giant_wide_summarize.py
         python scripts/distance_comparison_plot.py --render-only
         python scripts/giant_wide_validate.py ;;
 *) exit 2 ;;
esac
