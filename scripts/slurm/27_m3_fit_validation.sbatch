#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-m3-fit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
M23_STAGE="${M23_STAGE:-reference}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"
REPORT="${PROJECT_ROOT}/reports/generated/final/locked_four_task_m2_m3_v1/m3_${M23_STAGE}.json"
NOTEBOOK_OUTPUT="${PROJECT_ROOT}/outputs/private/executed_notebooks/09_m3_sequence_validation.ipynb"
if [[ "${M23_STAGE}" == "reference" ]]; then
  FINAL_NOTEBOOK_OUTPUT="outputs/private/executed_notebooks/09_m3_sequence_validation.ipynb"
else
  FINAL_NOTEBOOK_OUTPUT=""
fi

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize "final_m3_fit_${M23_STAGE}" "${SLURM_JOB_ID:?Slurm job ID required}"
export FINAL_M23_GIT_COMMIT="${FINAL_M1_GIT_COMMIT}"
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" m3-fit \
  --stage "${M23_STAGE}"
if [[ "${M23_STAGE}" == "reference" ]]; then
  final_container_python "${PYTHON}" scripts/execute_notebook.py \
    notebooks/09_m3_sequence_validation.ipynb "${NOTEBOOK_OUTPUT}"
  final_container_python "${PYTHON}" scripts/validate_executed_notebook.py "${NOTEBOOK_OUTPUT}"
  final_container_python "${PYTHON}" scripts/review_final_m23_results.py
fi
test -s "${REPORT}"
final_mark_success
