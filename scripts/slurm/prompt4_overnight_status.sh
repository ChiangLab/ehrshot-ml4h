#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
cd "${PROJECT_ROOT}"
ARMED_ENV="outputs/private/prompt4_overnight_armed_job_ids.env"
if [[ ! -s "${ARMED_ENV}" ]]; then
  echo "No Prompt 4 overnight submission record exists." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ARMED_ENV}"

printf 'Source commit: %s\n' "${SOURCE_GIT_COMMIT}"
printf 'Capacity job: %s\n' "${CAPACITY_JOB_ID}"
printf 'Launcher job: %s\n' "${LAUNCHER_JOB_ID}"

for job_id in "${CAPACITY_JOB_ID}" "${LAUNCHER_JOB_ID}"; do
  status_root="outputs/private/run_status/${job_id}"
  if [[ -s "${status_root}/FAILED.json" ]]; then
    printf 'FAILED: %s\n' "${job_id}"
  elif [[ -s "${status_root}/COMPLETE.json" ]]; then
    printf 'COMPLETE: %s\n' "${job_id}"
  elif squeue -h -j "${job_id}" 2>/dev/null | grep -q .; then
    printf 'ACTIVE_OR_PENDING: %s\n' "${job_id}"
  else
    printf 'NO_FINAL_SENTINEL: %s\n' "${job_id}"
  fi
done

printf '\n===== QUEUE =====\n'
squeue -u "${USER}" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"

LAUNCHED_ENV="outputs/private/prompt4_overnight_launched_job_ids.env"
if [[ -s "${LAUNCHED_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${LAUNCHED_ENV}"
  printf '\nM3 early reference report job: %s\n' "${M3_REFERENCE_REPORT_JOB_ID}"
  if [[ -s "outputs/private/run_status/${M3_REFERENCE_REPORT_JOB_ID}/FAILED.json" ]]; then
    printf 'FAILED: M3 early reference report\n'
  elif [[ -s "outputs/private/run_status/${M3_REFERENCE_REPORT_JOB_ID}/COMPLETE.json" ]]; then
    printf 'COMPLETE: M3 early reference report\n'
  elif squeue -h -j "${M3_REFERENCE_REPORT_JOB_ID}" 2>/dev/null | grep -q .; then
    printf 'ACTIVE_OR_PENDING: M3 early reference report\n'
  else
    printf 'NO_FINAL_SENTINEL: M3 early reference report\n'
  fi
  bash scripts/slurm/prompt4_m2_progress.sh || true
else
  printf '\nM2 has not been launched yet; inspect the capacity/launcher logs if both jobs left the queue.\n'
fi

FAILURES=0
for job_id in "${CAPACITY_JOB_ID}" "${LAUNCHER_JOB_ID}"; do
  [[ ! -s "outputs/private/run_status/${job_id}/FAILED.json" ]] || FAILURES=$((FAILURES + 1))
done
[[ "${FAILURES}" -eq 0 ]]
