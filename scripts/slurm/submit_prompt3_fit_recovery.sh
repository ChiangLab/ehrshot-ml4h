#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
RECOVERY_CONFIG="configs/recovery/prompt3_fit_after_446.yaml"
UPSTREAM_SOURCE_GIT_COMMIT="77c47f322b98fc74196cc56a0a16edc43604c887"
SOURCE_SMOKE_8B_JOB_ID="436"
SOURCE_FOUNDATION_JOB_ID="437"
SOURCE_THROUGHPUT_JOB_ID="444"
SOURCE_EMBED_ARRAY_JOB_ID="445"
SOURCE_FAILED_FIT_JOB_ID="446"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
EXECUTION_NODE="${TEMPORAL_EHR_EXECUTION_NODE:-dgit00038673}"

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing fit-only recovery submission from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ ! "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ || \
      "${SOURCE_GIT_COMMIT}" == "${UPSTREAM_SOURCE_GIT_COMMIT}" ]]; then
  echo "Fit-only recovery requires a new full validator commit after job 446." >&2
  exit 1
fi

FAILED_FIT_STATUS="outputs/private/run_status/${SOURCE_FAILED_FIT_JOB_ID}/FAILED.json"
for path in \
  "${RECOVERY_CONFIG}" \
  "outputs/private/run_status/${SOURCE_FOUNDATION_JOB_ID}/COMPLETE.json" \
  "outputs/private/run_status/${SOURCE_THROUGHPUT_JOB_ID}/COMPLETE.json" \
  "${FAILED_FIT_STATUS}" \
  reports/generated/final/locked_four_task_m1_v1/foundation.json \
  reports/generated/final/locked_four_task_m1_v1/throughput.json \
  "${MODEL_ROOT}/Qwen3-Embedding-0.6B/PINNED_REVISION.txt" \
  "${MODEL_ROOT}/Qwen3-Embedding-0.6B/MODEL_ARTIFACT_MANIFEST.json" \
  "${MODEL_ROOT}/Qwen3-Embedding-8B/PINNED_REVISION.txt" \
  "${MODEL_ROOT}/Qwen3-Embedding-8B/MODEL_ARTIFACT_MANIFEST.json"; do
  [[ -s "${path}" ]] || { echo "Missing fit-only recovery input: ${path}" >&2; exit 1; }
done
for index in 0 1 2 3; do
  [[ -s "outputs/private/run_status/${SOURCE_EMBED_ARRAY_JOB_ID}_${index}/COMPLETE.json" ]] \
    || { echo "Missing completed embedding sentinel for index ${index}." >&2; exit 1; }
done
for task in lab_anemia guo_readmission new_hypertension chexpert; do
  [[ -s "reports/generated/final/locked_four_task_m1_v1/embedding/${task}.json" ]] \
    || { echo "Missing embedding aggregate for ${task}." >&2; exit 1; }
done
for path in \
  outputs/private/final/locked_four_task_m1_v1/fit/validation_predictions.parquet \
  reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json \
  outputs/private/executed_notebooks/06_final_m1_results.ipynb; do
  [[ ! -e "${path}" ]] || { echo "Refusing existing partial/final fit output: ${path}" >&2; exit 1; }
done

FAILED_FIT_SENTINEL_SHA256="$(sha256sum "${FAILED_FIT_STATUS}" | cut -d' ' -f1)"
srun \
  --partition=batch \
  --nodelist="${EXECUTION_NODE}" \
  --job-name=tehr-prompt3-fit-recovery-preflight \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:20:00 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  env \
    PYTHONPATH=src \
    FINAL_M1_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
    FINAL_M1_FIT_RECOVERY_CONFIG="${RECOVERY_CONFIG}" \
    FINAL_M1_EXPECTED_FAILED_SENTINEL_SHA256="${FAILED_FIT_SENTINEL_SHA256}" \
    .venv/bin/python scripts/run_final_m1.py fit-reuse-gate

mkdir -p outputs/private/submissions
FIT_JOB_ID="$(
  sbatch \
    --parsable \
    --partition=batch \
    --export="ALL,FINAL_M1_FIT_RECOVERY_CONFIG=${RECOVERY_CONFIG},FINAL_M1_EXPECTED_FAILED_SENTINEL_SHA256=${FAILED_FIT_SENTINEL_SHA256},FINAL_STATUS_SOURCE_JOB_ID=${SOURCE_FAILED_FIT_JOB_ID},FINAL_STATUS_SOURCE_GIT_COMMIT=${UPSTREAM_SOURCE_GIT_COMMIT},FINAL_STATUS_RECOVERY_GIT_COMMIT=${SOURCE_GIT_COMMIT},FINAL_STATUS_SOURCE_FAILED_SENTINEL_SHA256=${FAILED_FIT_SENTINEL_SHA256}" \
    scripts/slurm/23_final_m1_fit_validation.sbatch
)"
FIT_JOB_ID="${FIT_JOB_ID%%;*}"
[[ "${FIT_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "Fit recovery returned invalid job ID: ${FIT_JOB_ID}" >&2; exit 1; }

SUBMISSION_FILE="outputs/private/submissions/prompt3-fit-recovery-${FIT_JOB_ID}.env"
[[ ! -e "${SUBMISSION_FILE}" ]] || { echo "Refusing to overwrite ${SUBMISSION_FILE}" >&2; exit 1; }
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'UPSTREAM_SOURCE_GIT_COMMIT=%q\n' "${UPSTREAM_SOURCE_GIT_COMMIT}"
  printf 'SMOKE_8B_JOB_ID=%q\n' "${SOURCE_SMOKE_8B_JOB_ID}"
  printf 'FOUNDATION_JOB_ID=%q\n' "${SOURCE_FOUNDATION_JOB_ID}"
  printf 'THROUGHPUT_JOB_ID=%q\n' "${SOURCE_THROUGHPUT_JOB_ID}"
  printf 'EMBED_ARRAY_JOB_ID=%q\n' "${SOURCE_EMBED_ARRAY_JOB_ID}"
  printf 'SOURCE_FAILED_FIT_JOB_ID=%q\n' "${SOURCE_FAILED_FIT_JOB_ID}"
  printf 'SOURCE_FAILED_FIT_SENTINEL_SHA256=%q\n' "${FAILED_FIT_SENTINEL_SHA256}"
  printf 'FIT_JOB_ID=%q\n' "${FIT_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
trap - EXIT
ln -sfn "submissions/prompt3-fit-recovery-${FIT_JOB_ID}.env" \
  outputs/private/prompt3_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'UPSTREAM_SOURCE_GIT_COMMIT=%s\n' "${UPSTREAM_SOURCE_GIT_COMMIT}"
printf 'PRESERVED_FAILED_FIT_JOB_ID=%s\n' "${SOURCE_FAILED_FIT_JOB_ID}"
printf 'REUSED_THROUGHPUT_JOB_ID=%s\n' "${SOURCE_THROUGHPUT_JOB_ID}"
printf 'REUSED_EMBED_ARRAY_JOB_ID=%s\n' "${SOURCE_EMBED_ARRAY_JOB_ID}"
printf 'FIT_JOB_ID=%s\n' "${FIT_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
