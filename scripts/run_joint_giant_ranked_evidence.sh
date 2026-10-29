#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 2
#SBATCH -t 01:00:00
#SBATCH --mem=8G
#SBATCH -o logs/joint_giant_ranked_evidence_%j.out
#SBATCH -e logs/joint_giant_ranked_evidence_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/joint_calibration_evidence_v2.py \
  --source data/joint_giant_ranked_v1/same_star.fits \
  --config configs/joint_giant_ranked_fields.json \
  --output data/joint_giant_ranked_v1_evidence_v2/same_star.fits \
  --manifest manifests/joint_giant_ranked_v1_evidence_v2.json
