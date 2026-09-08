#!/usr/bin/env bash
#SBATCH --job-name=tehr-m1-test-fit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m1_submission.yaml"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_m1_test_fit_report "${SLURM_JOB_ID:?Slurm job ID required}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
final_container_python "${PYTHON}" scripts/run_final_submission.py --config "${CONFIG}" fit-test-report
final_container_python "${PYTHON}" scripts/review_final_submission.py --config "${CONFIG}" --require-results
final_mark_success
