#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 04:00:00
#SBATCH -J baseline_v1_4_experiments
#SBATCH -o logs/baseline_v1_4_experiments_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
stage="${1:?Pass frozen, expanded, or matched}"
case "$stage" in frozen|expanded|matched) ;; *) exit 2 ;; esac
flock -n data/.baseline_v1_4_experiments.lock python -u scripts/baseline_v1_4_experiments.py "$stage"
