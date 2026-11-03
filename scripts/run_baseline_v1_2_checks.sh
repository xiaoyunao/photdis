#!/bin/bash
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH -J v12_checks
#SBATCH -o logs/baseline_v1_2_checks_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
pids=()
for name in A B C A_on_B A_on_C B_on_C; do
 python scripts/baseline_v1_2_score.py "$name" > "logs/baseline_v1_2_score_${name}_${SLURM_JOB_ID}.out" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
python scripts/baseline_v1_2_parallax.py
python scripts/baseline_v1_2_validate.py
python scripts/baseline_v1_2_report.py
