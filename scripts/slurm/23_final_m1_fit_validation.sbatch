#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-m1-fit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
FINAL_CONFIG="${PROJECT_ROOT}/configs/experiments/final_four_task_m1.yaml"
REPORT_PATH="${PROJECT_ROOT}/reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json"
NOTEBOOK_OUTPUT="${PROJECT_ROOT}/outputs/private/executed_notebooks/06_final_m1_results.ipynb"
FINAL_NOTEBOOK_OUTPUT="outputs/private/executed_notebooks/06_final_m1_results.ipynb"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_m1_fit_validation "${SLURM_JOB_ID:?Slurm job ID required}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
final_container_python "${PYTHON}" scripts/run_final_m1.py --config "${FINAL_CONFIG}" fit-report
final_container_python "${PYTHON}" scripts/execute_notebook.py \
  notebooks/06_final_m1_results.ipynb "${NOTEBOOK_OUTPUT}"
final_container_python "${PYTHON}" scripts/validate_executed_notebook.py "${NOTEBOOK_OUTPUT}"
final_container_python "${PYTHON}" scripts/review_final_m1_results.py \
  --report "${REPORT_PATH}" --private-postconditions
test -s "${REPORT_PATH}"
test -s "${NOTEBOOK_OUTPUT}"
final_mark_success
