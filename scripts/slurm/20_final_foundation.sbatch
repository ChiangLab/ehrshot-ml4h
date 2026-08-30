#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-foundation
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
FINAL_CONFIG="${PROJECT_ROOT}/configs/experiments/final_four_task_m1.yaml"
FOUNDATION_REPORT="${PROJECT_ROOT}/reports/generated/final/locked_four_task_m1_v1/foundation.json"
NOTEBOOK_OUTPUT="${PROJECT_ROOT}/outputs/private/executed_notebooks/05_final_foundation_review.ipynb"
FINAL_NOTEBOOK_OUTPUT="outputs/private/executed_notebooks/05_final_foundation_review.ipynb"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_foundation "${SLURM_JOB_ID:?Slurm job ID required}"

for path in "${EVENT_PATH}" "${FINAL_CONFIG}"; do
  [[ -s "${path}" ]] || { echo "Missing required file: ${path}" >&2; exit 1; }
done
for path in "${ASSET_ROOT}" "${MODEL_ROOT}/Qwen3-Embedding-0.6B"; do
  [[ -d "${path}" ]] || { echo "Missing required directory: ${path}" >&2; exit 1; }
done
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
final_container_python "${PYTHON}" scripts/run_final_m1.py --config "${FINAL_CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_m1.py --config "${FINAL_CONFIG}" foundation
final_container_python "${PYTHON}" scripts/review_final_foundation.py \
  --report "${FOUNDATION_REPORT}" --private-postconditions
final_container_python "${PYTHON}" scripts/execute_notebook.py \
  notebooks/05_final_foundation_review.ipynb "${NOTEBOOK_OUTPUT}"
final_container_python "${PYTHON}" scripts/validate_executed_notebook.py "${NOTEBOOK_OUTPUT}"
test -s "${FOUNDATION_REPORT}"
test -s "${NOTEBOOK_OUTPUT}"
final_mark_success
