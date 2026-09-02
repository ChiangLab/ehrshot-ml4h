#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
CONFIG="configs/experiments/final_m2_m3.yaml"
AUTHORIZATION="configs/recovery/prompt4_m2_after_502.yaml"
M3_SOURCE_COMMIT="1a97d8f1a70c47e18f96a2f56e63a370d6ff9913"
M3_PREP_ARRAY_JOB_ID=470
M3_ARRAY_JOB_ID=471

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Prompt 4 M2 recovery from a dirty worktree." >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ ! "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ || "${SOURCE_GIT_COMMIT}" == "${M3_SOURCE_COMMIT}" ]]; then
  echo "M2 recovery requires the committed post-job-502 hotfix." >&2
  exit 1
fi

export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export FINAL_M23_GIT_COMMIT="${SOURCE_GIT_COMMIT}"

RECOVERY_REPORT="reports/generated/final/locked_four_task_m2_m3_v1/m2_recovery_after_502.json"
[[ ! -e "${RECOVERY_REPORT}" ]] || {
  echo "Refusing to overwrite existing recovery report: ${RECOVERY_REPORT}" >&2
  exit 1
}

srun \
  --partition=batch \
  --nodelist=dgit00038673 \
  --job-name=tehr-p4-m2-recovery-preflight \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/validate_prompt4_m2_recovery.py \
    --config "${CONFIG}" \
    --authorization "${AUTHORIZATION}" \
    --output "${RECOVERY_REPORT}" \
    --live-endpoint-check
RECOVERY_REPORT_SHA256="$(sha256sum "${RECOVERY_REPORT}" | cut -d' ' -f1)"

M2_ARRAY_JOB_ID="$(
  sbatch --parsable \
    --array=0-3%1 \
    --export=ALL,M23_STAGE=reference \
    scripts/slurm/24_m2_summary_task_array.sbatch
)"
M2_ARRAY_JOB_ID="${M2_ARRAY_JOB_ID%%;*}"
M2_EMBED_ARRAY_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${M2_ARRAY_JOB_ID}" \
    --array=0-3%1 \
    --export=ALL,M23_STAGE=reference \
    scripts/slurm/24b_m2_embed_task_array.sbatch
)"
M2_EMBED_ARRAY_JOB_ID="${M2_EMBED_ARRAY_JOB_ID%%;*}"
M2_REVIEW_PREP_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${M2_ARRAY_JOB_ID}" \
    --export=ALL \
    scripts/slurm/25a_prepare_m2_fidelity_review.sbatch
)"
M2_REVIEW_PREP_JOB_ID="${M2_REVIEW_PREP_JOB_ID%%;*}"

mkdir -p outputs/private/submissions
SUBMISSION_FILE="outputs/private/submissions/prompt4-m2-recovery-${M2_ARRAY_JOB_ID}.env"
[[ ! -e "${SUBMISSION_FILE}" ]] || { echo "Refusing to overwrite ${SUBMISSION_FILE}" >&2; exit 1; }
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'REFERENCE_SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'PROMPT4_STAGE=%q\n' reference
  printf 'M2_ARRAY_JOB_ID=%q\n' "${M2_ARRAY_JOB_ID}"
  printf 'M2_EMBED_ARRAY_JOB_ID=%q\n' "${M2_EMBED_ARRAY_JOB_ID}"
  printf 'M2_REVIEW_PREP_JOB_ID=%q\n' "${M2_REVIEW_PREP_JOB_ID}"
  printf 'M3_PREP_ARRAY_JOB_ID=%q\n' "${M3_PREP_ARRAY_JOB_ID}"
  printf 'M3_ARRAY_JOB_ID=%q\n' "${M3_ARRAY_JOB_ID}"
  printf 'M3_EVIDENCE_SOURCE_GIT_COMMIT=%q\n' "${M3_SOURCE_COMMIT}"
  printf 'M2_RECOVERY_REPORT=%q\n' "${RECOVERY_REPORT}"
  printf 'M2_RECOVERY_REPORT_SHA256=%q\n' "${RECOVERY_REPORT_SHA256}"
  printf 'M2_SUMMARY_PARALLELISM=%q\n' 1
  printf 'M2_EMBED_PARALLELISM=%q\n' 1
  printf 'M3_PREPARE_PARALLELISM=%q\n' 1
  printf 'M3_TRAIN_PARALLELISM=%q\n' 2
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
trap - EXIT
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" \
  outputs/private/prompt4_reference_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'M2_ARRAY_JOB_ID=%s\n' "${M2_ARRAY_JOB_ID}"
printf 'M2_EMBED_ARRAY_JOB_ID=%s\n' "${M2_EMBED_ARRAY_JOB_ID}"
printf 'M2_REVIEW_PREP_JOB_ID=%s\n' "${M2_REVIEW_PREP_JOB_ID}"
printf 'REUSED_M3_PREP_ARRAY_JOB_ID=%s\n' "${M3_PREP_ARRAY_JOB_ID}"
printf 'REUSED_M3_ARRAY_JOB_ID=%s\n' "${M3_ARRAY_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
