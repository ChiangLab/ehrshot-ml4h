#!/usr/bin/env bash
#SBATCH --job-name=tehr-m2-selection-v3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --mail-user=wdelery@mednet.ucla.edu
#SBATCH --mail-type=FAIL,TIME_LIMIT_90
set -euo pipefail
umask 077
PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:?Project root required}"
CONTAINER_IMAGE=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
cd "$PROJECT_ROOT"
unset FINAL_M1_FROZEN_SOURCE_GIT_COMMIT FINAL_STATUS_SOURCE_GIT_COMMIT
unset FINAL_STATUS_RECOVERY_GIT_COMMIT FINAL_STATUS_SOURCE_JOB_ID
unset FINAL_STATUS_SOURCE_FAILED_SENTINEL_SHA256 FINAL_NOTEBOOK_OUTPUT
source scripts/slurm/final_job_common.sh
export M2_SELECTION_STATUS_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  M2_SELECTION_STATUS_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
final_initialize "m2_selection_v3_${M2_SELECTION_PHASE:?Phase required}" "$M2_SELECTION_STATUS_ID"
test "$(git rev-parse HEAD)" = "${M2_SELECTION_COMMIT:?Commit required}"
test -z "$(git status --porcelain)"

ARGS=("$M2_SELECTION_PHASE")
case "$M2_SELECTION_PHASE" in
  canary|confirmation) ;;
  readiness)
    ARGS+=(--canary-job "${M2_SELECTION_CANARY_JOB:?Canary required}")
    ARGS+=(--confirmation-job "${M2_SELECTION_CONFIRMATION_JOB:?Confirmation required}")
    ;;
  shard|embed) ARGS+=(--index "${SLURM_ARRAY_TASK_ID:?Array index required}") ;;
  seal)
    ARGS+=(--array "${M2_SELECTION_PARENT_ARRAY:?Parent array required}")
    ARGS+=(--readiness-job "${M2_SELECTION_READINESS_JOB:?Readiness required}")
    ;;
  fit) ARGS+=(--array "${M2_SELECTION_PARENT_ARRAY:?Parent array required}") ;;
  *) echo 'STOP: unsupported selection-v3 batch phase'; exit 1 ;;
esac
final_container_python "$PYTHON" scripts/run_m2_selection_recovery.py "${ARGS[@]}"
final_mark_success
