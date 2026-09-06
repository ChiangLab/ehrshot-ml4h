#!/usr/bin/env bash

set -euo pipefail

SUBMITTED_JOB_IDS=()
TEMPORARY_FILE=""
cleanup_partial_submission() {
  local exit_status=$?
  if [[ -n "${TEMPORARY_FILE}" ]]; then
    rm -f "${TEMPORARY_FILE}"
  fi
  if [[ "${exit_status}" -ne 0 && "${#SUBMITTED_JOB_IDS[@]}" -gt 0 ]]; then
    echo "Submission failed; canceling partial Prompt 4 chain: ${SUBMITTED_JOB_IDS[*]}" >&2
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
AUTHORIZATION="configs/recovery/prompt4_m2_fidelity_v5_after_536.yaml"
M3_SOURCE_COMMIT="1a97d8f1a70c47e18f96a2f56e63a370d6ff9913"
M3_PREP_ARRAY_JOB_ID=470
M3_ARRAY_JOB_ID=471
M2_SUMMARY_SHARDS_PER_TASK=8
M2_SUMMARY_PARALLELISM="${PROMPT4_M2_SUMMARY_PARALLELISM:-2}"
M2_EMBED_PARALLELISM="${PROMPT4_M2_EMBED_PARALLELISM:-1}"
SLURM_MAIL_USER="${PROMPT4_SLURM_MAIL_USER:-wdelery@mednet.ucla.edu}"

if [[ ! "${M2_SUMMARY_PARALLELISM}" =~ ^[12]$ ]]; then
  echo "PROMPT4_M2_SUMMARY_PARALLELISM must be 1 or 2." >&2
  exit 1
fi
if [[ ! "${M2_EMBED_PARALLELISM}" =~ ^[12]$ ]]; then
  echo "PROMPT4_M2_EMBED_PARALLELISM must be 1 or 2." >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing M2 fidelity-v5 recovery from a dirty worktree." >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ ! "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ || \
      "${SOURCE_GIT_COMMIT}" == "8682cb494b779efd0d6ea16785c08f61702bb4a0" ]]; then
  echo "M2 fidelity-v5 recovery requires its committed post-536 source." >&2
  exit 1
fi
if [[ -n "$(squeue -h -j 536 2>/dev/null)" ]]; then
  echo "Job 536 must leave the queue before fidelity-v5 preflight." >&2
  exit 1
fi

export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export FINAL_M23_GIT_COMMIT="${SOURCE_GIT_COMMIT}"
export M2_SUMMARY_SHARDS_PER_TASK

RECOVERY_REPORT="reports/generated/final/locked_four_task_m2_m3_v1/m2_fidelity_v5_recovery-${SOURCE_GIT_COMMIT}.json"
if [[ -e "${RECOVERY_REPORT}" ]]; then
  RECOVERY_REPORT="${RECOVERY_REPORT}" SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
    .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(Path(os.environ["RECOVERY_REPORT"]).read_text(encoding="utf-8"))
assert payload["status"] == "PASS"
assert payload["mode"] == "M2_FIDELITY_V5_EXTENDED_GENERATION_RECOVERY"
assert payload["source_git_commit"] == os.environ["SOURCE_GIT_COMMIT"]
assert payload["live_endpoint_check_passed"] is True
assert payload["live_synthetic_repetitions"] == 3
print("PASS: reusing immutable fidelity-v5 preflight for this source commit")
PY
else
  srun \
    --partition=batch \
    --nodelist=dgit00038673 \
    --job-name=tehr-p4-m2-fidelity-v5-preflight \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/data:/data \
    --container-workdir="${PROJECT_ROOT}" \
    --export=ALL \
    env PYTHONPATH=src .venv/bin/python scripts/validate_prompt4_m2_fidelity_v5_recovery.py \
      --config "${CONFIG}" \
      --authorization "${AUTHORIZATION}" \
      --output "${RECOVERY_REPORT}" \
      --live-endpoint-check
fi
RECOVERY_REPORT_SHA256="$(sha256sum "${RECOVERY_REPORT}" | cut -d' ' -f1)"

LAST_SHARD_INDEX=$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))
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
SUBMISSION_FILE="outputs/private/submissions/prompt4-m2-fidelity-v5-${M2_ARRAY_JOB_ID}.env"
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
  printf 'M2_EMBED_PARALLELISM=%q\n' "${M2_EMBED_PARALLELISM}"
  printf 'SLURM_MAIL_USER=%q\n' "${SLURM_MAIL_USER}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
TEMPORARY_FILE=""
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" \
  outputs/private/prompt4_reference_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'M2_SUMMARY_SHARD_ARRAY_JOB_ID=%s\n' "${M2_ARRAY_JOB_ID}"
printf 'M2_SUMMARY_SHARDS_PER_TASK=%s\n' "${M2_SUMMARY_SHARDS_PER_TASK}"
printf 'M2_EMBED_ARRAY_JOB_ID=%s\n' "${M2_EMBED_ARRAY_JOB_ID}"
printf 'M2_REVIEW_PREP_JOB_ID=%s\n' "${M2_REVIEW_PREP_JOB_ID}"
printf 'REUSED_M3_PREP_ARRAY_JOB_ID=%s\n' "${M3_PREP_ARRAY_JOB_ID}"
printf 'REUSED_M3_ARRAY_JOB_ID=%s\n' "${M3_ARRAY_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
trap - EXIT
