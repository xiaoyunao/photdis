#!/bin/bash
#SBATCH --job-name=photdis-faint-signed
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the Photdis checkout}"
export PYTHONPATH=scripts:src:.deps:.deps/v39_dust
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
runtime=/global/common/software/desi/perlmutter/desiconda/20260910-3.1.0/conda/bin/python
"$runtime" -m pytest -q tests/test_faint_signed.py

"$runtime" -u scripts/faint_signed_20261120.py pilot
"$runtime" -u scripts/faint_signed_20261120.py power
"$runtime" -u scripts/faint_signed_20261120.py full
"$runtime" -u scripts/faint_signed_20261120.py score
"$runtime" -u scripts/faint_signed_population.py
"$runtime" -u scripts/faint_signed_clusters.py
out=data/faint_signed_20261120
cp "$out/power/power.csv" "$out/results/"
"$runtime" -u scripts/faint_signed_figures.py "$out/results" "$out/figures" --regression
