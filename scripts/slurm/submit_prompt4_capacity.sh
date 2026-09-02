#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Prompt 4 capacity submission requires a clean DGX checkout." >&2
  exit 1
fi
export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
CAPACITY_SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
CAPACITY_JOB_ID="$(sbatch --parsable --export=ALL scripts/slurm/19_prompt4_capacity_probe.sbatch)"
mkdir -p outputs/private/submissions
SUBMISSION_FILE="outputs/private/submissions/prompt4-capacity-${CAPACITY_JOB_ID}.env"
{
  printf 'CAPACITY_SOURCE_GIT_COMMIT=%q\n' "${CAPACITY_SOURCE_GIT_COMMIT}"
  printf 'CAPACITY_JOB_ID=%q\n' "${CAPACITY_JOB_ID}"
} > "${SUBMISSION_FILE}"
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" outputs/private/prompt4_capacity_job_id.env
printf 'CAPACITY_SOURCE_GIT_COMMIT=%s\n' "${CAPACITY_SOURCE_GIT_COMMIT}"
printf 'CAPACITY_JOB_ID=%s\n' "${CAPACITY_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
