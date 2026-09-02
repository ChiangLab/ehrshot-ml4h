#!/usr/bin/env bash
#SBATCH --job-name=tehr-p4-capacity
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
REPORT="${PROJECT_ROOT}/reports/generated/engineering/prompt4_capacity.json"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize prompt4_synthetic_capacity "${SLURM_JOB_ID:?Slurm job ID required}"
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
free -h
lscpu | grep -E '^(CPU\(s\)|Model name|Thread|Core|Socket)'
final_container_python "${PYTHON}" scripts/dgx/benchmark_prompt4_capacity.py \
  --model-root "${MODEL_ROOT}" \
  --output "${REPORT}"
test -s "${REPORT}"
final_mark_success
