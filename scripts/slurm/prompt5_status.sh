#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
cd "${PROJECT_ROOT}"
STATUS_FILE="outputs/private/prompt5_job_ids.env"
[[ -s "${STATUS_FILE}" ]] || { echo "No Prompt 5 submission record: ${STATUS_FILE}" >&2; exit 1; }
source "${STATUS_FILE}"

printf 'Source commit: %s\n' "${SOURCE_GIT_COMMIT}"
printf 'Jobs: freeze=%s cohort=%s prepare=%s embed=%s fit=%s publication=%s\n' \
  "${FREEZE_JOB_ID}" "${COHORT_JOB_ID}" "${PREPARE_ARRAY_JOB_ID}" \
  "${EMBED_ARRAY_JOB_ID}" "${FIT_JOB_ID}" "${PUBLICATION_JOB_ID}"

echo "===== QUEUE ====="
squeue -j "${FREEZE_JOB_ID},${COHORT_JOB_ID},${PREPARE_ARRAY_JOB_ID},${EMBED_ARRAY_JOB_ID},${FIT_JOB_ID},${PUBLICATION_JOB_ID}" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R" || true

status_one() {
  local status_id="$1"
  if [[ -s "outputs/private/run_status/${status_id}/COMPLETE.json" && \
        ! -e "outputs/private/run_status/${status_id}/FAILED.json" ]]; then
    printf 'COMPLETE  %s\n' "${status_id}"
  elif [[ -s "outputs/private/run_status/${status_id}/FAILED.json" ]]; then
    printf 'FAILED    %s\n' "${status_id}"
  elif squeue -h -j "${status_id}" | grep -q .; then
    printf 'ACTIVE    %s\n' "${status_id}"
  else
    printf 'NO FINAL SENTINEL  %s\n' "${status_id}"
  fi
}

echo "===== SENTINELS ====="
status_one "${FREEZE_JOB_ID}"
status_one "${COHORT_JOB_ID}"
for index in 0 1 2 3; do status_one "${PREPARE_ARRAY_JOB_ID}_${index}"; done
for index in 0 1 2 3; do status_one "${EMBED_ARRAY_JOB_ID}_${index}"; done
status_one "${FIT_JOB_ID}"
status_one "${PUBLICATION_JOB_ID}"

echo "===== RECENT FAILURE LINES ====="
grep -HnE 'Traceback|ContractError|RuntimeError|Exited with exit code|FAILED|TIME LIMIT' \
  "slurm-tehr-m1-test-freeze-${FREEZE_JOB_ID}.out" \
  "slurm-tehr-m1-test-cohort-${COHORT_JOB_ID}.out" \
  slurm-tehr-m1-test-prepare-${PREPARE_ARRAY_JOB_ID}_*.out \
  slurm-tehr-m1-test-embed-${EMBED_ARRAY_JOB_ID}_*.out \
  "slurm-tehr-m1-test-fit-${FIT_JOB_ID}.out" \
  "slurm-tehr-m1-publication-${PUBLICATION_JOB_ID}.out" \
  2>/dev/null || echo "No obvious error lines found."
