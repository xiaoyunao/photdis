#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:30:00
#SBATCH --mem=8G
#SBATCH -o logs/joint_evidence_v2_%j.out
#SBATCH -e logs/joint_evidence_v2_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$1" == "v1" ]]; then
  config=configs/joint_calibration_fields.json
elif [[ "$1" == "targeted_v1" ]]; then
  config=configs/joint_targeted_fields.json
else
  echo "Unknown joint sample: $1" >&2
  exit 2
fi
python scripts/joint_calibration_evidence_v2.py \
  --source "data/joint_calibration_${1}/same_star.fits" \
  --config "$config" \
  --output "data/joint_calibration_${1}_evidence_v2/same_star.fits" \
  --manifest "manifests/joint_calibration_${1}_evidence_v2.json"
