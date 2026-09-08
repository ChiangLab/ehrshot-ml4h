#!/usr/bin/env bash

set -euo pipefail

CONFIRMATION="${1:-}"
EXPECTED_CONFIRMATION="I_CONFIRM_ONE_TIME_HELD_OUT_EVALUATION"
if [[ "${CONFIRMATION}" != "--confirm-test-evaluation=${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing held-out submission without the exact confirmation:" >&2
  echo "  --confirm-test-evaluation=${EXPECTED_CONFIRMATION}" >&2
  exit 2
fi

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
CONFIG="configs/experiments/final_m1_submission.yaml"

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing one-time held-out submission from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
[[ "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo "Invalid Git commit." >&2; exit 1; }

for path in \
  "${CONFIG}" \
  reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json \
  reports/generated/final/locked_four_task_m2_m3_v1/m3_reference.json \
  "${EVENT_PATH}" \
  "${MODEL_ROOT}/Qwen3-Embedding-0.6B/PINNED_REVISION.txt" \
  "${MODEL_ROOT}/Qwen3-Embedding-0.6B/MODEL_ARTIFACT_MANIFEST.json"; do
  [[ -s "${path}" ]] || { echo "Missing required frozen input: ${path}" >&2; exit 1; }
done
for path in \
  reports/generated/submission/locked_four_task_m1_submission_v1/FROZEN_EXPERIMENT_MANIFEST.json \
  outputs/private/submission/locked_four_task_m1_submission_v1/test_access/STARTED.json \
  reports/generated/submission/locked_four_task_m1_submission_v1/m1_held_out_results.json; do
  [[ ! -e "${path}" ]] || { echo "Refusing: one-time held-out state already exists: ${path}" >&2; exit 1; }
done

srun \
  --partition=batch \
  --nodelist=dgit00038673 \
  --job-name=tehr-m1-test-preflight \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:20:00 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  env PYTHONPATH=src FINAL_M1_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
  .venv/bin/python scripts/run_final_submission.py --config "${CONFIG}" hard-gates

submit_job() {
  local result
  result="$(sbatch --parsable "$@")"
  result="${result%%;*}"
  [[ "${result}" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: ${result}" >&2; exit 1; }
  printf '%s' "${result}"
}

FREEZE_JOB_ID="$(submit_job --export=ALL scripts/slurm/29_freeze_m1_submission.sbatch)"
COHORT_JOB_ID="$(submit_job --dependency="afterok:${FREEZE_JOB_ID}" --export=ALL scripts/slurm/30_build_m1_test_cohort.sbatch)"
PREPARE_ARRAY_JOB_ID="$(submit_job --dependency="afterok:${COHORT_JOB_ID}" --array=0-3%1 --export=ALL scripts/slurm/31_prepare_m1_test_task_array.sbatch)"
EMBED_ARRAY_JOB_ID="$(submit_job --dependency="afterok:${PREPARE_ARRAY_JOB_ID}" --array=0-3%1 --export=ALL scripts/slurm/32_embed_m1_test_task_array.sbatch)"
FIT_JOB_ID="$(submit_job --dependency="afterok:${EMBED_ARRAY_JOB_ID}" --export=ALL scripts/slurm/33_fit_m1_test_report.sbatch)"
PUBLICATION_JOB_ID="$(submit_job --dependency="afterok:${FIT_JOB_ID}" --export=ALL scripts/slurm/34_build_m1_publication_artifacts.sbatch)"

mkdir -p outputs/private/submissions
SUBMISSION_FILE="outputs/private/submissions/prompt5-m1-one-time-test-${FREEZE_JOB_ID}.env"
[[ ! -e "${SUBMISSION_FILE}" ]] || { echo "Refusing to overwrite ${SUBMISSION_FILE}" >&2; exit 1; }
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'FREEZE_JOB_ID=%q\n' "${FREEZE_JOB_ID}"
  printf 'COHORT_JOB_ID=%q\n' "${COHORT_JOB_ID}"
  printf 'PREPARE_ARRAY_JOB_ID=%q\n' "${PREPARE_ARRAY_JOB_ID}"
  printf 'EMBED_ARRAY_JOB_ID=%q\n' "${EMBED_ARRAY_JOB_ID}"
  printf 'FIT_JOB_ID=%q\n' "${FIT_JOB_ID}"
  printf 'PUBLICATION_JOB_ID=%q\n' "${PUBLICATION_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
trap - EXIT
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" outputs/private/prompt5_job_ids.env
cat "${SUBMISSION_FILE}"
