#!/usr/bin/env bash
#SBATCH --job-name=tehr-pilot-prepare
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
PILOT_CONFIG="${PROJECT_ROOT}/configs/experiments/complete_three_task_pilot.yaml"

source "${PROJECT_ROOT}/scripts/slurm/pilot_job_common.sh"
pilot_initialize pilot_prepare "${SLURM_JOB_ID:?Slurm job ID required}"

for path in "${EVENT_PATH}" "${PILOT_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "Missing required file: ${path##*/}" >&2; exit 1; }
done
for path in "${ASSET_ROOT}" "${MODEL_ROOT}/Qwen3-Embedding-0.6B"; do
  [[ -d "${path}" ]] || { echo "Missing required directory: ${path##*/}" >&2; exit 1; }
done

export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
pilot_container_python "${PYTHON}" scripts/run_pilot.py --config "${PILOT_CONFIG}" prepare
pilot_mark_success
