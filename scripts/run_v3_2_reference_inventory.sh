#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 00:45:00
#SBATCH -J photdis_v3_2_reference_inventory
#SBATCH -o logs/photdis_v3_2_reference_inventory_%j.out
set -euo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
set +u
source /global/common/software/desi/desi_environment.sh main
set -u
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$#" = 1 && "$1" =~ ^[0-9]+$ ]]; then
    receipt="logs/photdis_v3_2_reference_acquire_$1.out"
    root=data/v3_2/reference_sources/dr19_astra_0.6.0
    hashes=()
    for product in AstroNNdist ASPCAP; do
        name="$root/astraAllStar${product}-0.6.0.fits.gz"
        digest=$(awk -v path="$name" '$2==path && length($1)==64 && $1 ~ /^[0-9a-f]+$/ {print $1}' "$receipt")
        [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
        hashes+=("$digest")
    done
    set -- "${hashes[@]}"
fi
python -u scripts/v3_2_reference_inventory.py "$@"
