#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 00:30:00
#SBATCH -J baseline_v1_num
#SBATCH -o logs/baseline_v1_num_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/baseline_v1_sample.py
python scripts/baseline_v1_convergence.py A &
p1=$!
python scripts/baseline_v1_convergence.py B &
p2=$!
python scripts/baseline_v1_convergence.py C &
p3=$!
wait "$p1"
wait "$p2"
wait "$p3"
python scripts/baseline_v1_convergence.py finalize
