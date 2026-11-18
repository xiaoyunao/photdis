#!/bin/bash
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -q regular
#SBATCH -o logs/halo_%x_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/.deps:$PWD/.deps/v39_dust"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun -n1 -c "$SLURM_CPUS_PER_TASK" python "scripts/halo_$1.py" "${@:2}"
