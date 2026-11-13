#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --array=0-7%2
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_1_responserefit
#SBATCH -o logs/photdis_v3_1_responserefit_%A_%a.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for offset in 0 1 2 3; do
    draw=$((4 * SLURM_ARRAY_TASK_ID + offset))
    python -u scripts/v3_1_response_refit.py --draw "$draw" > "logs/photdis_v3_1_responserefit_${SLURM_ARRAY_JOB_ID}_draw${draw}.out" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
