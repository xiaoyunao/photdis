#!/bin/bash
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -t 02:00:00
#SBATCH -J photdis_v3_1_finalvalidation
#SBATCH -o logs/photdis_v3_1_finalvalidation_%j.out
set -eo pipefail
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/scripts:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
stage="${1:?admission, domain-prepare, freeze, evidence, score, report, domain-evidence or domain-assessment required}"
shift
case "$stage" in
  admission) python -u scripts/v3_1_validation_admission.py "$@" ;;
  domain-prepare) python -u scripts/v3_1_domain_prepare.py "$@" ;;
  freeze) python -u scripts/v3_1_final_freeze.py "$@" ;;
  evidence) python -u scripts/v3_1_validation_evidence.py "$@" ;;
  score) python -u scripts/v3_1_validation_score.py "$@" ;;
  report) python -u scripts/v3_1_validation_report.py "$@" ;;
  domain-evidence) python -u scripts/v3_1_domain_evidence.py "$@" ;;
  domain-assessment) python -u scripts/v3_1_domain_assessment.py "$@" ;;
  *) echo 'Unknown validation stage' >&2; exit 2 ;;
esac
