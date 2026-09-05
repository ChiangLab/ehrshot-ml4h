#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
cd "${PROJECT_ROOT}"

printf '===== SOURCE =====\n'
git status --short --branch
printf 'HEAD=%s\n' "$(git rev-parse HEAD)"

printf '\n===== LIVE SLURM JOBS =====\n'
squeue -u "${USER}" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R" || true

for environment_file in \
  outputs/private/prompt4_capacity_job_id.env \
  outputs/private/prompt4_reference_job_ids.env \
  outputs/private/prompt4_reference_finalize_job_ids.env \
  outputs/private/prompt4_sensitivity_job_ids.env; do
  if [[ -f "${environment_file}" ]]; then
    # shellcheck disable=SC1090
    source "${environment_file}"
  fi
done

declare -a STATUS_IDS=()
declare -A SEEN=()

append_single() {
  local value="${1:-}"
  if [[ -n "${value}" && -z "${SEEN[${value}]:-}" ]]; then
    STATUS_IDS+=("${value}")
    SEEN["${value}"]=1
  fi
}

append_array() {
  local value="${1:-}" maximum="${2}"
  if [[ -n "${value}" ]]; then
    local index
    for index in $(seq 0 "${maximum}"); do
      append_single "${value}_${index}"
    done
  fi
}

append_single "${CAPACITY_JOB_ID:-}"
if [[ -n "${M2_ARRAY_JOB_ID:-}" ]]; then
  if [[ -n "${M2_SUMMARY_SHARDS_PER_TASK:-}" ]]; then
    append_array "${M2_ARRAY_JOB_ID}" "$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))"
  else
    append_array "${M2_ARRAY_JOB_ID}" 3
  fi
fi
append_array "${M2_EMBED_ARRAY_JOB_ID:-}" 3
append_single "${M2_REVIEW_PREP_JOB_ID:-}"
append_array "${M3_PREP_ARRAY_JOB_ID:-}" 3
append_array "${M3_ARRAY_JOB_ID:-}" 11
append_single "${M2_FIT_JOB_ID:-}"
append_single "${M3_FIT_JOB_ID:-}"
if [[ -n "${SENSITIVITY_SUMMARY_ARRAY_JOB_ID:-}" ]]; then
  if [[ -n "${M2_SUMMARY_SHARDS_PER_TASK:-}" ]]; then
    append_array "${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}" "$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))"
  else
    append_array "${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}" 3
  fi
fi
append_array "${SENSITIVITY_EMBED_ARRAY_JOB_ID:-}" 3
append_array "${SENSITIVITY_M3_PREP_ARRAY_JOB_ID:-}" 3
if [[ -n "${SENSITIVITY_M3_PREP_ARRAY_JOB_ID:-}" ]]; then
  append_array "${SENSITIVITY_ARRAY_JOB_ID:-}" 11
fi
append_single "${SENSITIVITY_FIT_JOB_ID:-}"
append_single "${FREEZE_JOB_ID:-}"
for status_id in "$@"; do append_single "${status_id}"; done

printf '\n===== DURABLE PROMPT 4 STATUS =====\n'
if [[ "${#STATUS_IDS[@]}" -eq 0 ]]; then
  echo "No Prompt 4 submission records were found."
else
  for status_id in "${STATUS_IDS[@]}"; do
    status_dir="outputs/private/run_status/${status_id}"
    queue_state="$(squeue -h -j "${status_id}" -o '%T' 2>/dev/null | head -n 1 || true)"
    if [[ -s "${status_dir}/FAILED.json" ]]; then
      final_state="FAILED"
    elif [[ -s "${status_dir}/COMPLETE.json" ]]; then
      final_state="COMPLETE"
    elif [[ -n "${queue_state}" ]]; then
      final_state="${queue_state}"
    elif [[ -s "${status_dir}/RUNNING.json" ]]; then
      final_state="ENDED_WITHOUT_FINAL_SENTINEL"
    else
      final_state="NO_SENTINEL"
    fi
    printf '%-14s %s\n' "${status_id}" "${final_state}"
  done
fi

printf '\n===== RECENT PROMPT 4 LOG FILES =====\n'
find . -maxdepth 1 -type f \
  \( -name 'slurm-tehr-p4-capacity-*.out' \
     -o -name 'slurm-tehr-final-m2-*.out' \
     -o -name 'slurm-tehr-m2-review-packet-*.out' \
     -o -name 'slurm-tehr-final-m3-*.out' \
     -o -name 'slurm-tehr-freeze-prompt4-*.out' \) \
  -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s bytes %f\n' 2>/dev/null | sort | tail -n 30
