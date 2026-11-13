#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --array=0-25%4
#SBATCH -t 01:00:00
#SBATCH -J photdis_v3_1_heldouttransfer
#SBATCH -o logs/photdis_v3_1_heldouttransfer_%A_%a.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cases=(broad candidate holdout)
pids=()
for offset in 0 1 2 3; do
    index=$((4 * SLURM_ARRAY_TASK_ID + offset))
    if (( index >= 102 )); then continue; fi
    case_index=$((index / 34))
    correction=$((index % 34))
    python -u scripts/v3_1_heldout_transfer.py --case "${cases[$case_index]}" --correction "$correction" > "logs/photdis_v3_1_heldouttransfer_${SLURM_ARRAY_JOB_ID}_${index}.out" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
