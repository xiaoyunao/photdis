#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_3_sed
#SBATCH -o logs/photdis_v3_3_sed_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
phase="$1"
shift
if [[ "$phase" == 'fit' ]]; then
    set -- --partition "$SLURM_ARRAY_TASK_ID" "$@"
fi
mkdir -p data/v3_3/sed/initial
exec 9>"data/v3_3/sed/initial/.${phase}-${SLURM_ARRAY_TASK_ID:-single}.lock"
flock -n 9
python -u scripts/v3_3_sed.py "$phase" "$@"
