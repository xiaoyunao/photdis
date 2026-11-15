#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --mem=200G
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_5_dr11_production
#SBATCH -o logs/photdis_v3_5_dr11_production_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == setup || "$1" == smoke ]]; then
    python -u scripts/v3_5_dr11_production.py "$@"
else
    export SLURM_CPUS_PER_TASK=32
    pids=()
    for lane in 0 1 2 3; do
        srun --exclusive --exact -N 1 -n 1 -c 32 --mem=48G python -u scripts/v3_5_dr11_production.py run "$SLURM_ARRAY_TASK_ID" "$lane" 4 &
        pids+=("$!")
    done
    result=0
    for pid in "${pids[@]}"; do
        wait "$pid" || result=1
    done
    exit "$result"
fi
