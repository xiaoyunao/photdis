#!/bin/bash
#SBATCH -J p37_diagnostics
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 120
#SBATCH -t 01:15:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=data/v3_7
mkdir -p "$out/logs"

step() {
    local log="$1" cpus="$2" mem="$3"
    shift 3
    srun --exact -N1 -n1 -c "$cpus" --mem "$mem" env SLURM_CPUS_PER_TASK="$cpus" python "$@" \
        > "$out/logs/$log-$SLURM_JOB_ID.log" 2>&1
}

pids=()
for run in classifier_sky blue_neighborhood lf_known_branches use_domains; do
    if [ -f "$out/$run/complete.json" ]; then
        continue
    fi
    case "$run" in
        classifier_sky) cpus=8; mem=28G ;;
        blue_neighborhood) cpus=4; mem=16G ;;
        lf_known_branches) cpus=12; mem=16G ;;
        use_domains) cpus=4; mem=18G ;;
    esac
    step "$run" "$cpus" "$mem" "scripts/v3_7_$run.py" &
    pids+=("$!")
done

selection() {
    local program=scripts/v3_7_selection_replay.py
    if [ -f "$out/selection/complete.json" ]; then
        return
    fi
    if [ ! -f "$out/selection/prepare.json" ]; then
        step selection_prepare 4 20G "$program" prepare --shards 12 || return
    fi
    local shards=() error=0 shard pid marker
    for shard in {0..11}; do
        printf -v marker '%s/selection/replay_%02d.json' "$out" "$shard"
        if [ ! -f "$marker" ]; then
            step "selection_$shard" 7 12G "$program" replay --shard "$shard" &
            shards+=("$!")
        fi
    done
    for pid in "${shards[@]}"; do
        wait "$pid" || error=1
    done
    if [ "$error" -ne 0 ]; then
        return 1
    fi
    step selection_summary 4 24G "$program" summarize
}

selection &
pids+=("$!")
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
exit "$failed"
