#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Prompt 4 M2 capacity submission requires a clean DGX checkout." >&2
  exit 1
fi
if squeue -h -j 559,560,561 2>/dev/null | grep -q .; then
  echo "Jobs 559, 560, and 561 must leave the queue before the capacity ladder." >&2
  exit 1
fi
export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
CAPACITY_JOB_ID="$(
  sbatch --parsable --export=ALL scripts/slurm/19b_prompt4_m2_capacity_v7.sbatch
)"
CAPACITY_JOB_ID="${CAPACITY_JOB_ID%%;*}"
mkdir -p outputs/private/submissions
SUBMISSION_FILE="outputs/private/submissions/prompt4-m2-capacity-v7-${CAPACITY_JOB_ID}.env"
{
  printf 'CAPACITY_SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'CAPACITY_JOB_ID=%q\n' "${CAPACITY_JOB_ID}"
  printf 'CAPACITY_REPORT=%q\n' \
    "reports/generated/engineering/prompt4_m2_capacity_v7-${CAPACITY_JOB_ID}.json"
} > "${SUBMISSION_FILE}"
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" \
  outputs/private/prompt4_m2_capacity_v7_job_id.env
printf 'CAPACITY_SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'CAPACITY_JOB_ID=%s\n' "${CAPACITY_JOB_ID}"
printf 'CAPACITY_REPORT=%s\n' \
  "reports/generated/engineering/prompt4_m2_capacity_v7-${CAPACITY_JOB_ID}.json"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
