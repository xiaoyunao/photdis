#!/bin/bash
#SBATCH -J p38_figures
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
run() {
    if [ ! -f "data/v3_8/direct_figures/$1/complete.json" ]; then
        srun --exact -N1 -n1 -c 12 --mem 24G python scripts/v3_8_direct_figures.py "$1" \
            > "logs/p38_$1-$SLURM_JOB_ID.log" 2>&1
    fi
}
pids=()
for mode in reference overview sky; do
    run "$mode" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
