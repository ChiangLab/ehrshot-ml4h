#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-m3-prepare
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --array=0-3%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
M23_STAGE="${M23_STAGE:-reference}"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
STATUS_ID="${SLURM_ARRAY_JOB_ID:?Array job ID required}_${ARRAY_INDEX}"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize "final_m3_prepare_${M23_STAGE}" "${STATUS_ID}"
export FINAL_M23_GIT_COMMIT="${FINAL_M1_GIT_COMMIT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" m3-prepare-task \
  --task-index "${ARRAY_INDEX}" --stage "${M23_STAGE}"
final_mark_success
