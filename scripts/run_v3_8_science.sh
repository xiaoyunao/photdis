#!/bin/bash
#SBATCH -J p38_science
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH -t 00:45:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for mode in reliability branches north; do
    if [ ! -f "data/v3_8/$mode/complete.json" ]; then
        srun --exact -N1 -n1 -c 12 --mem 24G python "scripts/v3_8_$mode.py" \
            > "logs/p38_$mode-$SLURM_JOB_ID.log" 2>&1 &
        pids+=("$!")
    fi
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
