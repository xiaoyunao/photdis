#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:15:00
#SBATCH --mem=2G
#SBATCH -o logs/joint_poor_anchor_audit_%j.out
#SBATCH -e logs/joint_poor_anchor_audit_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/audit_joint_poor_anchors.py \
  --source data/joint_calibration_targeted_v1/same_star.fits \
  --output manifests/joint_targeted_poor_anchors.json
