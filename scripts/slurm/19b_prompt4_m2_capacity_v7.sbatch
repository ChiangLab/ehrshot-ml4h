#!/usr/bin/env bash
#SBATCH --job-name=tehr-p4-m2-cap-v7
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="configs/experiments/final_m2_m3.yaml"
REPORT="${PROJECT_ROOT}/reports/generated/engineering/prompt4_m2_capacity_v7-${SLURM_JOB_ID}.json"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize prompt4_m2_synthetic_capacity_v7 "${SLURM_JOB_ID:?Slurm job ID required}"
free -h
lscpu | grep -E '^(CPU\(s\)|Model name|Thread|Core|Socket)'
final_container_python "${PYTHON}" scripts/dgx/benchmark_prompt4_m2_capacity_v7.py \
  --config "${CONFIG}" \
  --output "${REPORT}" \
  --source-git-commit "${FINAL_M1_GIT_COMMIT}"
test -s "${REPORT}"
final_mark_success
