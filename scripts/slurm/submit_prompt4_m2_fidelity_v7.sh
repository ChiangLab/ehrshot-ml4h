#!/usr/bin/env bash

set -euo pipefail

SUBMITTED_JOB_IDS=()
TEMPORARY_FILE=""
cleanup_partial_submission() {
  local exit_status=$?
  [[ -z "${TEMPORARY_FILE}" ]] || rm -f "${TEMPORARY_FILE}"
  if [[ "${exit_status}" -ne 0 && "${#SUBMITTED_JOB_IDS[@]}" -gt 0 ]]; then
    echo "Submission failed; canceling partial fidelity-v7 chain: ${SUBMITTED_JOB_IDS[*]}" >&2
    scancel "${SUBMITTED_JOB_IDS[@]}" 2>/dev/null || true
  fi
  trap - EXIT
  exit "${exit_status}"
}
trap cleanup_partial_submission EXIT

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
CONFIG="configs/experiments/final_m2_m3.yaml"
AUTHORIZATION="configs/recovery/prompt4_m2_fidelity_v7_after_559.yaml"
M3_SOURCE_COMMIT="1a97d8f1a70c47e18f96a2f56e63a370d6ff9913"
M3_PREP_ARRAY_JOB_ID=470
M3_ARRAY_JOB_ID=471
M2_SUMMARY_SHARDS_PER_TASK=192
M2_REQUESTS_PER_SHARD=8
M2_EMBED_PARALLELISM="${PROMPT4_M2_EMBED_PARALLELISM:-1}"
SLURM_MAIL_USER="${PROMPT4_SLURM_MAIL_USER:-wdelery@mednet.ucla.edu}"

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing M2 fidelity-v7 recovery from a dirty worktree." >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if squeue -h -j 559,560,561 2>/dev/null | grep -q .; then
  echo "Jobs 559, 560, and 561 must leave the queue before fidelity-v7 recovery." >&2
  exit 1
fi
if [[ ! "${M2_EMBED_PARALLELISM}" =~ ^[12]$ ]]; then
  echo "PROMPT4_M2_EMBED_PARALLELISM must be 1 or 2." >&2
  exit 1
fi
CAPACITY_ENV="outputs/private/prompt4_m2_capacity_v7_job_id.env"
if [[ ! -s "${CAPACITY_ENV}" ]]; then
  echo "Run submit_prompt4_m2_capacity_v7.sh and wait for its COMPLETE sentinel first." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${CAPACITY_ENV}"
if [[ "${CAPACITY_SOURCE_GIT_COMMIT}" != "${SOURCE_GIT_COMMIT}" ]]; then
  echo "Capacity evidence and recovery source commits differ." >&2
  exit 1
fi
if [[ ! -s "outputs/private/run_status/${CAPACITY_JOB_ID}/COMPLETE.json" || \
      -e "outputs/private/run_status/${CAPACITY_JOB_ID}/FAILED.json" || \
      ! -s "${CAPACITY_REPORT}" ]]; then
  echo "The fidelity-v7 capacity job is not durably COMPLETE." >&2
  exit 1
fi

M2_GLOBAL_CONCURRENCY="$(
  CAPACITY_REPORT="${CAPACITY_REPORT}" .venv/bin/python -c \
    'import json, os; print(json.load(open(os.environ["CAPACITY_REPORT"]))["recommended_global_concurrency"])'
)"
case "${M2_GLOBAL_CONCURRENCY}" in
  8|16|24|32) ;;
  *) echo "Capacity report contains an unsupported recommendation." >&2; exit 1 ;;
esac
M2_SUMMARY_PARALLELISM=$((M2_GLOBAL_CONCURRENCY / M2_REQUESTS_PER_SHARD))

export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export FINAL_M23_GIT_COMMIT="${SOURCE_GIT_COMMIT}"
export M2_SUMMARY_SHARDS_PER_TASK

RECOVERY_REPORT="reports/generated/final/locked_four_task_m2_m3_v1/m2_fidelity_v7_recovery-${SOURCE_GIT_COMMIT}.json"
if [[ -e "${RECOVERY_REPORT}" ]]; then
  RECOVERY_REPORT="${RECOVERY_REPORT}" SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
    CAPACITY_JOB_ID="${CAPACITY_JOB_ID}" .venv/bin/python - <<'PY'
import json
import os

payload = json.load(open(os.environ["RECOVERY_REPORT"]))
assert payload["status"] == "PASS"
assert payload["mode"] == "M2_FIDELITY_V7_CAPACITY_SHARDED_RECOVERY"
assert payload["source_git_commit"] == os.environ["SOURCE_GIT_COMMIT"]
assert payload["capacity_job_id"] == os.environ["CAPACITY_JOB_ID"]
print("PASS: reusing immutable fidelity-v7 recovery report")
PY
else
  srun \
    --partition=batch \
    --nodelist=dgit00038673 \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/data:/data \
    --container-workdir="${PROJECT_ROOT}" \
    --export=ALL \
    env PYTHONPATH=src .venv/bin/python scripts/validate_prompt4_m2_fidelity_v7_recovery.py \
      --config "${CONFIG}" \
      --authorization "${AUTHORIZATION}" \
      --capacity-report "${CAPACITY_REPORT}" \
      --capacity-job-id "${CAPACITY_JOB_ID}" \
      --output "${RECOVERY_REPORT}"
fi
RECOVERY_REPORT_SHA256="$(sha256sum "${RECOVERY_REPORT}" | cut -d' ' -f1)"
CAPACITY_REPORT_SHA256="$(sha256sum "${CAPACITY_REPORT}" | cut -d' ' -f1)"

LAST_SHARD_INDEX=$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))
MAX_ARRAY_SIZE="$(scontrol show config | awk '$1 == "MaxArraySize" {print $3; exit}')"
if [[ ! "${MAX_ARRAY_SIZE}" =~ ^[1-9][0-9]*$ || \
      "${LAST_SHARD_INDEX}" -ge "${MAX_ARRAY_SIZE}" ]]; then
  echo "Slurm MaxArraySize=${MAX_ARRAY_SIZE:-unknown} cannot admit indices 0-${LAST_SHARD_INDEX}." >&2
  exit 1
fi
MAIL_ARGS=(
  --mail-user="${SLURM_MAIL_USER}"
  --mail-type=END,FAIL,TIME_LIMIT_80,TIME_LIMIT_90
)
M2_ARRAY_JOB_ID="$(
  sbatch --parsable \
    "${MAIL_ARGS[@]}" \
    --array="0-${LAST_SHARD_INDEX}%${M2_SUMMARY_PARALLELISM}" \
    --export=ALL,M23_STAGE=reference,M2_SUMMARY_SHARDS_PER_TASK="${M2_SUMMARY_SHARDS_PER_TASK}" \
    scripts/slurm/24c_m2_summary_shard_array.sbatch
)"
M2_ARRAY_JOB_ID="${M2_ARRAY_JOB_ID%%;*}"
SUBMITTED_JOB_IDS+=("${M2_ARRAY_JOB_ID}")
M2_EMBED_ARRAY_JOB_ID="$(
  sbatch --parsable \
    "${MAIL_ARGS[@]}" \
    --dependency="afterok:${M2_ARRAY_JOB_ID}" \
    --array="0-3%${M2_EMBED_PARALLELISM}" \
    --export=ALL,M23_STAGE=reference \
    scripts/slurm/24b_m2_embed_task_array.sbatch
)"
M2_EMBED_ARRAY_JOB_ID="${M2_EMBED_ARRAY_JOB_ID%%;*}"
SUBMITTED_JOB_IDS+=("${M2_EMBED_ARRAY_JOB_ID}")
M2_REVIEW_PREP_JOB_ID="$(
  sbatch --parsable \
    "${MAIL_ARGS[@]}" \
    --dependency="afterok:${M2_ARRAY_JOB_ID}" \
    --export=ALL \
    scripts/slurm/25a_prepare_m2_fidelity_review.sbatch
)"
M2_REVIEW_PREP_JOB_ID="${M2_REVIEW_PREP_JOB_ID%%;*}"
SUBMITTED_JOB_IDS+=("${M2_REVIEW_PREP_JOB_ID}")

mkdir -p outputs/private/submissions
SUBMISSION_FILE="outputs/private/submissions/prompt4-m2-fidelity-v7-${M2_ARRAY_JOB_ID}.env"
[[ ! -e "${SUBMISSION_FILE}" ]] || {
  echo "Refusing to overwrite ${SUBMISSION_FILE}" >&2
  exit 1
}
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
{
  printf 'REFERENCE_SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'PROMPT4_STAGE=%q\n' reference
  printf 'M2_ARRAY_JOB_ID=%q\n' "${M2_ARRAY_JOB_ID}"
  printf 'M2_SUMMARY_SHARD_ARRAY_JOB_ID=%q\n' "${M2_ARRAY_JOB_ID}"
  printf 'M2_SUMMARY_SHARDS_PER_TASK=%q\n' "${M2_SUMMARY_SHARDS_PER_TASK}"
  printf 'M2_EMBED_ARRAY_JOB_ID=%q\n' "${M2_EMBED_ARRAY_JOB_ID}"
  printf 'M2_REVIEW_PREP_JOB_ID=%q\n' "${M2_REVIEW_PREP_JOB_ID}"
  printf 'M3_PREP_ARRAY_JOB_ID=%q\n' "${M3_PREP_ARRAY_JOB_ID}"
  printf 'M3_ARRAY_JOB_ID=%q\n' "${M3_ARRAY_JOB_ID}"
  printf 'M3_EVIDENCE_SOURCE_GIT_COMMIT=%q\n' "${M3_SOURCE_COMMIT}"
  printf 'M2_RECOVERY_REPORT=%q\n' "${RECOVERY_REPORT}"
  printf 'M2_RECOVERY_REPORT_SHA256=%q\n' "${RECOVERY_REPORT_SHA256}"
  printf 'M2_SUMMARY_PARALLELISM=%q\n' "${M2_SUMMARY_PARALLELISM}"
  printf 'M2_REQUESTS_PER_SHARD=%q\n' "${M2_REQUESTS_PER_SHARD}"
  printf 'M2_MAXIMUM_TOTAL_ENDPOINT_REQUESTS=%q\n' "${M2_GLOBAL_CONCURRENCY}"
  printf 'M2_EMBED_PARALLELISM=%q\n' "${M2_EMBED_PARALLELISM}"
  printf 'CAPACITY_JOB_ID=%q\n' "${CAPACITY_JOB_ID}"
  printf 'CAPACITY_REPORT=%q\n' "${CAPACITY_REPORT}"
  printf 'CAPACITY_REPORT_SHA256=%q\n' "${CAPACITY_REPORT_SHA256}"
  printf 'SLURM_MAIL_USER=%q\n' "${SLURM_MAIL_USER}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
TEMPORARY_FILE=""
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" \
  outputs/private/prompt4_reference_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'CAPACITY_JOB_ID=%s\n' "${CAPACITY_JOB_ID}"
printf 'SELECTED_GLOBAL_CONCURRENCY=%s\n' "${M2_GLOBAL_CONCURRENCY}"
printf 'M2_SUMMARY_SHARD_ARRAY_JOB_ID=%s\n' "${M2_ARRAY_JOB_ID}"
printf 'M2_SUMMARY_SHARDS_PER_TASK=%s\n' "${M2_SUMMARY_SHARDS_PER_TASK}"
printf 'M2_SUMMARY_PARALLELISM=%s\n' "${M2_SUMMARY_PARALLELISM}"
printf 'M2_EMBED_ARRAY_JOB_ID=%s\n' "${M2_EMBED_ARRAY_JOB_ID}"
printf 'M2_REVIEW_PREP_JOB_ID=%s\n' "${M2_REVIEW_PREP_JOB_ID}"
printf 'REUSED_M3_PREP_ARRAY_JOB_ID=%s\n' "${M3_PREP_ARRAY_JOB_ID}"
printf 'REUSED_M3_ARRAY_JOB_ID=%s\n' "${M3_ARRAY_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
printf 'MONITOR_COMMAND=%s\n' "bash scripts/slurm/prompt4_m2_progress.sh"
trap - EXIT
