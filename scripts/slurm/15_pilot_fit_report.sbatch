#!/usr/bin/env bash
#SBATCH --job-name=tehr-pilot-report
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PILOT_CONFIG="${PROJECT_ROOT}/configs/experiments/complete_three_task_pilot.yaml"
REPORT_PATH="${PROJECT_ROOT}/reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"
NOTEBOOK_OUTPUT="${PROJECT_ROOT}/outputs/private/executed_notebooks/04_pilot_results.ipynb"
PILOT_NOTEBOOK_OUTPUT="outputs/private/executed_notebooks/04_pilot_results.ipynb"

source "${PROJECT_ROOT}/scripts/slurm/pilot_job_common.sh"
pilot_initialize pilot_fit_report "${SLURM_JOB_ID:?Slurm job ID required}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
pilot_container_python "${PYTHON}" scripts/run_pilot.py --config "${PILOT_CONFIG}" fit-report
pilot_container_python "${PYTHON}" scripts/execute_notebook.py \
  notebooks/04_pilot_results.ipynb "${NOTEBOOK_OUTPUT}"
pilot_container_python "${PYTHON}" scripts/validate_executed_notebook.py "${NOTEBOOK_OUTPUT}"
pilot_container_python "${PYTHON}" scripts/review_pilot_results.py --report "${REPORT_PATH}"
test -s "${REPORT_PATH}"
test -s "${NOTEBOOK_OUTPUT}"
pilot_mark_success
