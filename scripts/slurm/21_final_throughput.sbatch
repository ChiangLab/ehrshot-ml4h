#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-throughput
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
FINAL_CONFIG="${PROJECT_ROOT}/configs/experiments/final_four_task_m1.yaml"
THROUGHPUT_REPORT="${PROJECT_ROOT}/reports/generated/final/locked_four_task_m1_v1/throughput.json"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_throughput "${SLURM_JOB_ID:?Slurm job ID required}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
final_container_python "${PYTHON}" scripts/run_final_m1.py --config "${FINAL_CONFIG}" throughput
test -s "${THROUGHPUT_REPORT}"
final_mark_success
