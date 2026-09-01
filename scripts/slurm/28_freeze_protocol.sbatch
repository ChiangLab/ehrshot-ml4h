#!/usr/bin/env bash
#SBATCH --job-name=tehr-freeze-prompt4
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"
MANIFEST="${PROJECT_ROOT}/reports/generated/final/locked_four_task_m2_m3_v1/FROZEN_EXPERIMENT_MANIFEST.json"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_prompt4_freeze "${SLURM_JOB_ID:?Slurm job ID required}"
export FINAL_M23_GIT_COMMIT="${FINAL_M1_GIT_COMMIT}"
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" freeze
final_container_python "${PYTHON}" scripts/review_final_m23_results.py --require-frozen
HANDOFF_ARGS=(--output reports/generated/handoff/prompt4_handoff.json)
IFS=',' read -r -a STATUS_IDS <<< "${PROMPT4_STATUS_IDS:-}"
for status_id in "${STATUS_IDS[@]}"; do
  [[ -n "${status_id}" ]] && HANDOFF_ARGS+=(--job-id "${status_id}")
done
final_container_python "${PYTHON}" scripts/build_handoff_bundle.py "${HANDOFF_ARGS[@]}"
test -s "${MANIFEST}"
test -s "${PROJECT_ROOT}/reports/generated/handoff/prompt4_handoff.json"
final_mark_success
