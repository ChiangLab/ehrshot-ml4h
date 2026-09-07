#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
cd "${PROJECT_ROOT}"

ENVIRONMENT_FILE="outputs/private/prompt4_reference_job_ids.env"
if [[ ! -s "${ENVIRONMENT_FILE}" ]]; then
  echo "No active Prompt 4 reference submission record was found." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${ENVIRONMENT_FILE}"

SHARDS_PER_TASK="${M2_SUMMARY_SHARDS_PER_TASK:-192}"
TOTAL_SHARDS=$((4 * SHARDS_PER_TASK))
SUMMARY_COMPLETE=0
SUMMARY_FAILED=0
SUMMARY_OTHER=0
for index in $(seq 0 $((TOTAL_SHARDS - 1))); do
  STATUS_ROOT="outputs/private/run_status/${M2_ARRAY_JOB_ID}_${index}"
  if [[ -s "${STATUS_ROOT}/FAILED.json" ]]; then
    SUMMARY_FAILED=$((SUMMARY_FAILED + 1))
  elif [[ -s "${STATUS_ROOT}/COMPLETE.json" ]]; then
    SUMMARY_COMPLETE=$((SUMMARY_COMPLETE + 1))
  else
    SUMMARY_OTHER=$((SUMMARY_OTHER + 1))
  fi
done

EMBED_COMPLETE=0
EMBED_FAILED=0
EMBED_OTHER=0
for index in 0 1 2 3; do
  STATUS_ROOT="outputs/private/run_status/${M2_EMBED_ARRAY_JOB_ID}_${index}"
  if [[ -s "${STATUS_ROOT}/FAILED.json" ]]; then
    EMBED_FAILED=$((EMBED_FAILED + 1))
  elif [[ -s "${STATUS_ROOT}/COMPLETE.json" ]]; then
    EMBED_COMPLETE=$((EMBED_COMPLETE + 1))
  else
    EMBED_OTHER=$((EMBED_OTHER + 1))
  fi
done

REVIEW_STATE="NOT_STARTED"
REVIEW_ROOT="outputs/private/run_status/${M2_REVIEW_PREP_JOB_ID}"
if [[ -s "${REVIEW_ROOT}/FAILED.json" ]]; then
  REVIEW_STATE="FAILED"
elif [[ -s "${REVIEW_ROOT}/COMPLETE.json" ]]; then
  REVIEW_STATE="COMPLETE"
elif squeue -h -j "${M2_REVIEW_PREP_JOB_ID}" 2>/dev/null | grep -q .; then
  REVIEW_STATE="ACTIVE_OR_PENDING"
fi

printf 'Source commit: %s\n' "${REFERENCE_SOURCE_GIT_COMMIT}"
printf 'Summary shards: complete=%s failed=%s active_or_pending=%s total=%s\n' \
  "${SUMMARY_COMPLETE}" "${SUMMARY_FAILED}" "${SUMMARY_OTHER}" "${TOTAL_SHARDS}"
printf 'Embedding tasks: complete=%s failed=%s active_or_pending=%s total=4\n' \
  "${EMBED_COMPLETE}" "${EMBED_FAILED}" "${EMBED_OTHER}"
printf 'Manual-review preparation: %s\n' "${REVIEW_STATE}"

printf '\n===== LIVE JOBS =====\n'
squeue -j "${M2_ARRAY_JOB_ID},${M2_EMBED_ARRAY_JOB_ID},${M2_REVIEW_PREP_JOB_ID}" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R" || true

printf '\n===== LATEST AGGREGATE-SAFE PROGRESS =====\n'
if compgen -G "slurm-tehr-final-m2-shard-${M2_ARRAY_JOB_ID}_*.out" >/dev/null; then
  grep '"event": "m2_summary_progress"' \
    slurm-tehr-final-m2-shard-${M2_ARRAY_JOB_ID}_*.out 2>/dev/null | tail -n 20 || true
else
  echo "No summary-shard logs exist yet."
fi

if [[ "${SUMMARY_FAILED}" -gt 0 || "${EMBED_FAILED}" -gt 0 || "${REVIEW_STATE}" == "FAILED" ]]; then
  exit 1
fi
