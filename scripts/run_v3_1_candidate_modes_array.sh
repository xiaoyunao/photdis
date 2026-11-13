#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH --array=0-7%2
#SBATCH -t 01:00:00
#SBATCH -J photdis_v3_1_candidatemodesfull
#SBATCH -o logs/photdis_v3_1_candidatemodesfull_%A_%a.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
bash scripts/run_v3_1_candidate_modes.sh --shard "${SLURM_ARRAY_TASK_ID:?}" --shards 8 --workers 32
