#!/usr/bin/env bash
#SBATCH --job-name=tehr-pilot-throughput
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PILOT_CONFIG="${PROJECT_ROOT}/configs/experiments/complete_three_task_pilot.yaml"

source "${PROJECT_ROOT}/scripts/slurm/pilot_job_common.sh"
pilot_initialize pilot_throughput "${SLURM_JOB_ID:?Slurm job ID required}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
pilot_container_python "${PYTHON}" scripts/run_pilot.py --config "${PILOT_CONFIG}" throughput
test -s "${PROJECT_ROOT}/reports/generated/pilot/three_task_m1_engineering_pilot_v1/throughput.json"
pilot_mark_success
