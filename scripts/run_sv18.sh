#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o logs/sv18_%x_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="${PHOTDIS_SV18_SCRIPTS:-$PWD/scripts}:$PWD/src:$PWD/scripts:$PWD/.deps:$PWD/.deps/v39_dust"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun -n1 -c "$SLURM_CPUS_PER_TASK" python "${PHOTDIS_SV18_SCRIPTS:-scripts}/sv18_$1.py" "${@:2}"
