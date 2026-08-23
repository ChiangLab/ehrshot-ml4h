#!/usr/bin/env bash
#SBATCH --job-name=tehr-pilot-embed
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --partition=batch
#SBATCH --array=0-2%1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PILOT_CONFIG="${PROJECT_ROOT}/configs/experiments/complete_three_task_pilot.yaml"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
STATUS_ID="${SLURM_ARRAY_JOB_ID:?Array job ID required}_${ARRAY_INDEX}"

source "${PROJECT_ROOT}/scripts/slurm/pilot_job_common.sh"
pilot_initialize pilot_embed_task "${STATUS_ID}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
pilot_container_python "${PYTHON}" scripts/run_pilot.py \
  --config "${PILOT_CONFIG}" embed-task --task-index "${ARRAY_INDEX}"
pilot_mark_success
