#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 04:00:00
#SBATCH -J baseline_v1_4_controls
#SBATCH -o logs/baseline_v1_4_controls_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec 9>data/.baseline_v1_4_experiments.lock
flock -n 9
exec 8>data/.baseline_v1_4_gaia_refine.lock
flock -n 8
python -c 'import json; from pathlib import Path; p=Path("data/baseline_v1_4"); a=json.loads((p/"assembly.json").read_text()); b=json.loads((p/"gaia_refined/frozen_complete.json").read_text()); assert a["reference_sha256"]==b["reference_sha256"]'
for stage in expanded matched; do
    python -u scripts/baseline_v1_4_experiments.py "$stage"
    python -u scripts/baseline_v1_4_gaia_refine.py "$stage"
done
python -u scripts/baseline_v1_4_validate.py
