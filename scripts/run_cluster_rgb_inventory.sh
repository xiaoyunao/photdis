#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:45:00
#SBATCH --mem=8G
#SBATCH -o logs/cluster_rgb_inventory_%A_%a.out
#SBATCH -e logs/cluster_rgb_inventory_%A_%a.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
    python scripts/cluster_rgb_inventory.py m3
elif [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then
    python scripts/cluster_rgb_inventory.py ngc2419
else
    echo "Unknown cluster array index: $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
