#!/usr/bin/env bash
#SBATCH --job-name=tehr-final-m2-shard
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --array=0-767%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --mail-user=wdelery@mednet.ucla.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80,TIME_LIMIT_90

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
M23_STAGE="${M23_STAGE:-reference}"
SHARDS_PER_TASK="${M2_SUMMARY_SHARDS_PER_TASK:-192}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"
ARRAY_INDEX="${SLURM_ARRAY_TASK_ID:?Array task ID required}"
STATUS_ID="${SLURM_ARRAY_JOB_ID:?Array job ID required}_${ARRAY_INDEX}"

if [[ "${SHARDS_PER_TASK}" != "192" || "${ARRAY_INDEX}" -ge 768 ]]; then
  echo "Prompt 4 accelerated recovery requires 192 shards for each of four tasks." >&2
  exit 1
fi
# Interleave tasks so the first active wave exercises all four task families.
TASK_INDEX=$((ARRAY_INDEX % 4))
SHARD_INDEX=$((ARRAY_INDEX / 4))

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize "final_m2_summary_shard_${M23_STAGE}" "${STATUS_ID}"
export FINAL_M23_GIT_COMMIT="${FINAL_M1_GIT_COMMIT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" m2-task \
  --task-index "${TASK_INDEX}" \
  --stage "${M23_STAGE}" \
  --summaries-only \
  --summary-shard-index "${SHARD_INDEX}" \
  --summary-shard-count "${SHARDS_PER_TASK}"
final_mark_success
