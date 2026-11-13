#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --array=2-31%3
#SBATCH -t 03:00:00
#SBATCH -J photdis_v3_1_structureentryresume
#SBATCH -o logs/photdis_v3_1_structureentryresume_%A_%a.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
bash scripts/run_v3_1_structure_entry_resume.sh --draw "${SLURM_ARRAY_TASK_ID:?}" --workers 32
