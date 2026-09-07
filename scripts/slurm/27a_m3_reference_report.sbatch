#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-m3-report
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"
REPORT="${PROJECT_ROOT}/reports/generated/final/locked_four_task_m2_m3_v1/m3_reference.json"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_m3_reference_report_early "${SLURM_JOB_ID:?Slurm job ID required}"
export FINAL_M23_GIT_COMMIT="${FINAL_M1_GIT_COMMIT}"
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" \
  m3-reference-report
test -s "${REPORT}"
final_mark_success
