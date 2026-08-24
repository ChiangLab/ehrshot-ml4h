#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
MODEL_DIR="${MODEL_ROOT}/Qwen3-Embedding-0.6B"

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Prompt 2 submission from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi

SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ ! "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Prompt 2 requires a full source Git commit." >&2
  exit 1
fi

for path in \
  "${EVENT_PATH}" \
  "${MODEL_DIR}/PINNED_REVISION.txt" \
  "${MODEL_DIR}/MODEL_ARTIFACT_MANIFEST.json" \
  "outputs/private/cohorts/primary_panel.parquet" \
  "reports/generated/count_baseline/validation_results.json" \
  "configs/experiments/complete_three_task_pilot.yaml" \
  "configs/experiments/locked_panel.yaml"; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing required Prompt 2 input: ${path}" >&2
    exit 1
  fi
done

if [[ ! -d "${ASSET_ROOT}" ]]; then
  echo "Missing EHRSHOT asset directory: ${ASSET_ROOT}" >&2
  exit 1
fi

for task in lab_anemia new_acutemi guo_los; do
  prediction_path="outputs/private/count_baseline/${task}/validation_predictions.parquet"
  if [[ ! -s "${prediction_path}" ]]; then
    echo "Missing governed count-baseline predictions for ${task}." >&2
    exit 1
  fi
done

mkdir -p outputs/private/submissions
SMOKE_JOB_ID="$(
  sbatch --parsable --partition=batch --export=ALL scripts/slurm/03_smoke_qwen_06b.sbatch
)"
SMOKE_JOB_ID="${SMOKE_JOB_ID%%;*}"
if [[ ! "${SMOKE_JOB_ID}" =~ ^[0-9]+$ ]]; then
  echo "Smoke submission did not return one numeric Slurm job ID." >&2
  exit 1
fi

CHAIN_OUTPUT="$(
  PILOT_AFTEROK_JOB_ID="${SMOKE_JOB_ID}" bash scripts/slurm/submit_pilot_chain.sh
)"
eval "${CHAIN_OUTPUT}"

SUBMISSION_FILE="outputs/private/submissions/prompt2-${SMOKE_JOB_ID}.env"
if [[ -e "${SUBMISSION_FILE}" ]]; then
  echo "Refusing to overwrite existing submission record: ${SUBMISSION_FILE}" >&2
  exit 1
fi
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'SMOKE_JOB_ID=%q\n' "${SMOKE_JOB_ID}"
  printf '%s\n' "${CHAIN_OUTPUT}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
trap - EXIT
ln -sfn "submissions/prompt2-${SMOKE_JOB_ID}.env" outputs/private/prompt2_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'SMOKE_JOB_ID=%s\n' "${SMOKE_JOB_ID}"
printf '%s\n' "${CHAIN_OUTPUT}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
