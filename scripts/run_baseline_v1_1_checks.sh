#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --mem=256G
#SBATCH -t 02:00:00
#SBATCH -J baseline_v1_1_checks
#SBATCH -o logs/baseline_v1_1_checks_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/baseline_v1_1_score.py
pids=()
for kind in parent parallax; do
  for name in A B C; do
    python scripts/baseline_v1_1_sample_predict.py "$kind" "$name" > "logs/baseline_v1_1_${kind}_${name}_${SLURM_JOB_ID}.out" 2>&1 &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" != 0 ]]; then exit 1; fi
python scripts/baseline_v1_1_risk.py
python scripts/baseline_v1_1_parallax.py
