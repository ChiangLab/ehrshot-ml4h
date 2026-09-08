#!/usr/bin/env bash
#SBATCH --job-name=tehr-m1-test-cohort
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m1_submission.yaml"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_m1_test_cohort "${SLURM_JOB_ID:?Slurm job ID required}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
final_container_python "${PYTHON}" scripts/run_final_submission.py --config "${CONFIG}" test-cohort
final_container_python "${PYTHON}" scripts/review_final_submission.py --config "${CONFIG}" --require-test-cohort
final_mark_success
