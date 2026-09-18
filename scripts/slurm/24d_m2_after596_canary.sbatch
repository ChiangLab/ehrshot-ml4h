#!/usr/bin/env bash
#SBATCH --job-name=tehr-m2-recurrence-canary
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --mail-user=wdelery@mednet.ucla.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90

set -euo pipefail
umask 077
PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:?Project root required}"
export TEMPORAL_EHR_MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Model root required}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
cd "${PROJECT_ROOT}"
test "$(git rev-parse HEAD)" = "${M2_CANARY_SOURCE_COMMIT:?Source commit required}"
test -z "$(git status --porcelain)"
unset FINAL_M1_FROZEN_SOURCE_GIT_COMMIT FINAL_STATUS_SOURCE_GIT_COMMIT
unset FINAL_STATUS_RECOVERY_GIT_COMMIT FINAL_STATUS_SOURCE_JOB_ID
unset FINAL_STATUS_SOURCE_FAILED_SENTINEL_SHA256 FINAL_NOTEBOOK_OUTPUT
source scripts/slurm/final_job_common.sh
final_initialize m2_after596_canary "${SLURM_JOB_ID:?Job ID required}"
final_container_python "${PYTHON}" scripts/run_m2_after596_canary.py run --job-id "${SLURM_JOB_ID}"
final_mark_success
