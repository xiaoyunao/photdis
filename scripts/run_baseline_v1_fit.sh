#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J baseline_v1_fit
#SBATCH -o logs/baseline_v1_fit_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for name in A B C A_on_B A_on_C B_on_C; do
  python scripts/baseline_v1_fit.py "$1" "$name" > "logs/baseline_v1_${1}_${name}_${SLURM_JOB_ID}.out" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
