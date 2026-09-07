#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
SLURM_MAIL_USER="${PROMPT4_SLURM_MAIL_USER:-wdelery@mednet.ucla.edu}"

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Overnight Prompt 4 submission requires a clean DGX checkout." >&2
  exit 1
fi
if squeue -h -j 559,560,561 2>/dev/null | grep -q .; then
  echo "Stop jobs 559-561 and wait for them to leave the queue first." >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"

ARMED_ENV="outputs/private/prompt4_overnight_armed_job_ids.env"
if [[ -s "${ARMED_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ARMED_ENV}"
  if [[ "${SOURCE_GIT_COMMIT}" == "$(git rev-parse HEAD)" ]] && {
       squeue -h -j "${CAPACITY_JOB_ID},${LAUNCHER_JOB_ID}" 2>/dev/null | grep -q . ||
       [[ -s "outputs/private/run_status/${LAUNCHER_JOB_ID}/COMPLETE.json" &&
          ! -e "outputs/private/run_status/${LAUNCHER_JOB_ID}/FAILED.json" ]];
     }; then
    echo "Prompt 4 overnight automation is already armed for this source commit."
    printf 'CAPACITY_JOB_ID=%s\n' "${CAPACITY_JOB_ID}"
    printf 'LAUNCHER_JOB_ID=%s\n' "${LAUNCHER_JOB_ID}"
    exit 0
  fi
fi

CAPACITY_ENV="outputs/private/prompt4_m2_capacity_v7_job_id.env"
REUSE_CAPACITY=0
if [[ -s "${CAPACITY_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CAPACITY_ENV}"
  if [[ "${CAPACITY_SOURCE_GIT_COMMIT:-}" == "${SOURCE_GIT_COMMIT}" ]] && {
       squeue -h -j "${CAPACITY_JOB_ID}" 2>/dev/null | grep -q . ||
       [[ -s "outputs/private/run_status/${CAPACITY_JOB_ID}/COMPLETE.json" &&
          ! -e "outputs/private/run_status/${CAPACITY_JOB_ID}/FAILED.json" ]];
     }; then
    REUSE_CAPACITY=1
  fi
fi
if [[ "${REUSE_CAPACITY}" -eq 0 ]]; then
  bash scripts/slurm/submit_prompt4_m2_capacity_v7.sh
  # shellcheck disable=SC1090
  source "${CAPACITY_ENV}"
fi

DEPENDENCY_ARGS=()
if squeue -h -j "${CAPACITY_JOB_ID}" 2>/dev/null | grep -q .; then
  DEPENDENCY_ARGS+=(--dependency="afterany:${CAPACITY_JOB_ID}")
fi
LAUNCHER_JOB_ID="$(
  sbatch --parsable \
    "${DEPENDENCY_ARGS[@]}" \
    --mail-user="${SLURM_MAIL_USER}" \
    --mail-type=END,FAIL \
    --export=ALL,PROMPT4_CAPACITY_JOB_ID="${CAPACITY_JOB_ID}" \
    scripts/slurm/19c_launch_prompt4_m2_overnight_v7.sbatch
)"
LAUNCHER_JOB_ID="${LAUNCHER_JOB_ID%%;*}"

mkdir -p outputs/private/submissions
SUBMISSION_FILE="outputs/private/submissions/prompt4-overnight-armed-${LAUNCHER_JOB_ID}.env"
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'CAPACITY_JOB_ID=%q\n' "${CAPACITY_JOB_ID}"
  printf 'LAUNCHER_JOB_ID=%q\n' "${LAUNCHER_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" \
  outputs/private/prompt4_overnight_armed_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'CAPACITY_JOB_ID=%s\n' "${CAPACITY_JOB_ID}"
printf 'LAUNCHER_JOB_ID=%s\n' "${LAUNCHER_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
printf 'STATUS_COMMAND=%s\n' "bash scripts/slurm/prompt4_overnight_status.sh"
