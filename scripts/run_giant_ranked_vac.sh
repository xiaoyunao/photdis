#!/bin/bash
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -t 00:20:00
#SBATCH --mem=6G
#SBATCH -o logs/giant_ranked_vac_%j.out
#SBATCH -e logs/giant_ranked_vac_%j.err

set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/giant_vac_inventory.py \
  --source data/joint_giant_ranked_v1_evidence_v2/same_star.fits \
  --output data/giant_ranked_vac_v1/same_star.fits \
  --manifest manifests/giant_ranked_vac_v1.json
