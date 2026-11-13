#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH -J photdis_v3_1_radialpilots
#SBATCH -o logs/photdis_v3_1_radialpilots_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for catalog in 48 112; do
    python -u scripts/v3_1_radial_likelihood.py --catalog "$catalog" --step .01 --audit-selection --workers 32
    for arm in broad local; do
        python -u scripts/v3_1_radial_recovery.py --catalog "$catalog" --arm "$arm"
        python -u scripts/v3_1_radial_quadrature.py --catalog "$catalog" --arm "$arm"
    done
done
