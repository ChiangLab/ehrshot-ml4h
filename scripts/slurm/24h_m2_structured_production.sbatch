#!/usr/bin/env bash
#SBATCH --job-name=tehr-m2-production
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
export M2_PRODUCTION_STATUS_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  M2_PRODUCTION_STATUS_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
fi
final_initialize "m2_structured_production_${M2_PRODUCTION_PHASE:?Phase required}" "$M2_PRODUCTION_STATUS_ID"
test "$(git rev-parse HEAD)" = "${M2_PRODUCTION_COMMIT:?Commit required}"
test -z "$(git status --porcelain)"
ARGS=("$M2_PRODUCTION_PHASE")
case "$M2_PRODUCTION_PHASE" in
  shard|embed) ARGS+=(--index "${SLURM_ARRAY_TASK_ID:?Array index required}") ;;
  seal|fit) ARGS+=(--array "${M2_PRODUCTION_PARENT_ARRAY:?Parent array required}") ;;
  *) echo 'STOP: unsupported batch phase'; exit 1 ;;
esac
final_container_python "$PYTHON" scripts/run_m2_structured_production.py "${ARGS[@]}"
final_mark_success
