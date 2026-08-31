#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
SOURCE_FOUNDATION_JOB_ID="${SOURCE_FOUNDATION_JOB_ID:-437}"
SOURCE_SMOKE_8B_JOB_ID="${SOURCE_SMOKE_8B_JOB_ID:-436}"
SOURCE_FAILED_THROUGHPUT_JOB_ID="${SOURCE_FAILED_THROUGHPUT_JOB_ID:-438}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
EXECUTION_NODE="${TEMPORAL_EHR_EXECUTION_NODE:-dgit00038673}"

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Prompt 3 recovery submission from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ ! "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Prompt 3 recovery requires a full source Git commit." >&2
  exit 1
fi

FOUNDATION_STATUS="outputs/private/run_status/${SOURCE_FOUNDATION_JOB_ID}/COMPLETE.json"
FAILED_THROUGHPUT_STATUS="outputs/private/run_status/${SOURCE_FAILED_THROUGHPUT_JOB_ID}/FAILED.json"
FOUNDATION_REPORT="reports/generated/final/locked_four_task_m1_v1/foundation.json"
SMOKE_REPORT="outputs/private/smoke/qwen_embedding_smoke-${SOURCE_SMOKE_8B_JOB_ID}.json"
for path in \
  "${FOUNDATION_STATUS}" \
  "${FAILED_THROUGHPUT_STATUS}" \
  "${FOUNDATION_REPORT}" \
  "${SMOKE_REPORT}" \
  "${MODEL_ROOT}/Qwen3-Embedding-0.6B/PINNED_REVISION.txt" \
  "${MODEL_ROOT}/Qwen3-Embedding-0.6B/MODEL_ARTIFACT_MANIFEST.json" \
  "${MODEL_ROOT}/Qwen3-Embedding-8B/PINNED_REVISION.txt" \
  "${MODEL_ROOT}/Qwen3-Embedding-8B/MODEL_ARTIFACT_MANIFEST.json"; do
  [[ -s "${path}" ]] || { echo "Missing required Prompt 3 recovery input: ${path}" >&2; exit 1; }
done
if [[ -e reports/generated/final/locked_four_task_m1_v1/throughput.json ]]; then
  echo "Refusing to overwrite an existing throughput aggregate." >&2
  exit 1
fi

srun \
  --partition=batch \
  --nodelist="${EXECUTION_NODE}" \
  --job-name=tehr-prompt3-recovery-preflight \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:15:00 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  env PYTHONPATH=src FINAL_M1_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
    .venv/bin/python scripts/run_final_m1.py foundation-reuse-gate
srun \
  --partition=batch \
  --nodelist="${EXECUTION_NODE}" \
  --job-name=tehr-prompt3-foundation-review \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/review_final_foundation.py \
    --report "${FOUNDATION_REPORT}" --private-postconditions

mkdir -p outputs/private/submissions
THROUGHPUT_JOB_ID="$(
  sbatch --parsable --partition=batch --export=ALL scripts/slurm/21_final_throughput.sbatch
)"
THROUGHPUT_JOB_ID="${THROUGHPUT_JOB_ID%%;*}"
EMBED_ARRAY_JOB_ID="$(
  sbatch --parsable --partition=batch --dependency="afterok:${THROUGHPUT_JOB_ID}" --export=ALL \
    scripts/slurm/22_final_m1_embed_task_array.sbatch
)"
EMBED_ARRAY_JOB_ID="${EMBED_ARRAY_JOB_ID%%;*}"
FIT_JOB_ID="$(
  sbatch --parsable --partition=batch --dependency="afterok:${EMBED_ARRAY_JOB_ID}" --export=ALL \
    scripts/slurm/23_final_m1_fit_validation.sbatch
)"
FIT_JOB_ID="${FIT_JOB_ID%%;*}"
for job_id in "${THROUGHPUT_JOB_ID}" "${EMBED_ARRAY_JOB_ID}" "${FIT_JOB_ID}"; do
  [[ "${job_id}" =~ ^[0-9]+$ ]] || { echo "Recovery submission returned invalid job ID: ${job_id}" >&2; exit 1; }
done

FAILED_THROUGHPUT_SENTINEL_SHA256="$(sha256sum "${FAILED_THROUGHPUT_STATUS}" | cut -d' ' -f1)"
SUBMISSION_FILE="outputs/private/submissions/prompt3-recovery-${THROUGHPUT_JOB_ID}.env"
[[ ! -e "${SUBMISSION_FILE}" ]] || { echo "Refusing to overwrite ${SUBMISSION_FILE}" >&2; exit 1; }
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'SMOKE_8B_JOB_ID=%q\n' "${SOURCE_SMOKE_8B_JOB_ID}"
  printf 'FOUNDATION_JOB_ID=%q\n' "${SOURCE_FOUNDATION_JOB_ID}"
  printf 'SOURCE_FAILED_THROUGHPUT_JOB_ID=%q\n' "${SOURCE_FAILED_THROUGHPUT_JOB_ID}"
  printf 'SOURCE_FAILED_THROUGHPUT_SENTINEL_SHA256=%q\n' "${FAILED_THROUGHPUT_SENTINEL_SHA256}"
  printf 'THROUGHPUT_JOB_ID=%q\n' "${THROUGHPUT_JOB_ID}"
  printf 'EMBED_ARRAY_JOB_ID=%q\n' "${EMBED_ARRAY_JOB_ID}"
  printf 'FIT_JOB_ID=%q\n' "${FIT_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
trap - EXIT
ln -sfn "submissions/prompt3-recovery-${THROUGHPUT_JOB_ID}.env" \
  outputs/private/prompt3_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'REUSED_FOUNDATION_JOB_ID=%s\n' "${SOURCE_FOUNDATION_JOB_ID}"
printf 'PRESERVED_FAILED_THROUGHPUT_JOB_ID=%s\n' "${SOURCE_FAILED_THROUGHPUT_JOB_ID}"
printf 'THROUGHPUT_JOB_ID=%s\n' "${THROUGHPUT_JOB_ID}"
printf 'EMBED_ARRAY_JOB_ID=%s\n' "${EMBED_ARRAY_JOB_ID}"
printf 'FIT_JOB_ID=%s\n' "${FIT_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
