#!/usr/bin/env bash

set -euo pipefail

CONFIRMATION="${1:-}"
EXPECTED_CONFIRMATION="I_CONFIRM_RESUME_SAME_FROZEN_HELD_OUT_EVALUATION"
if [[ "${CONFIRMATION}" != "--confirm-resume=${EXPECTED_CONFIRMATION}" ]]; then
  echo "Refusing recovery without the exact same-manifest confirmation:" >&2
  echo "  --confirm-resume=${EXPECTED_CONFIRMATION}" >&2
  exit 2
fi

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before recovery}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before recovery}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before recovery}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
CONFIG="configs/experiments/final_m1_submission.yaml"
MANIFEST="reports/generated/submission/locked_four_task_m1_submission_v1/FROZEN_EXPERIMENT_MANIFEST.json"
RESULT="reports/generated/submission/locked_four_task_m1_submission_v1/m1_held_out_results.json"
PUBLICATION="reports/generated/submission/locked_four_task_m1_submission_v1/publication/PUBLICATION_COMPLETE.json"
STATUS_FILE="outputs/private/prompt5_job_ids.env"
RECOVERY_CONFIG="configs/recovery/prompt5_prepare_after_574.yaml"

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing same-manifest recovery from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
[[ -s "${MANIFEST}" ]] || { echo "No frozen Prompt 5 manifest exists; recovery is not applicable." >&2; exit 1; }
[[ -s "${STATUS_FILE}" ]] || { echo "Missing original Prompt 5 submission record." >&2; exit 1; }

source "${STATUS_FILE}"
ORIGINAL_FREEZE_JOB_ID="${FREEZE_JOB_ID}"
ORIGINAL_COHORT_JOB_ID="${COHORT_JOB_ID}"
ORIGINAL_PREPARE_ARRAY_JOB_ID="${PREPARE_ARRAY_JOB_ID}"
ORIGINAL_EMBED_ARRAY_JOB_ID="${EMBED_ARRAY_JOB_ID}"
ORIGINAL_FIT_JOB_ID="${FIT_JOB_ID}"
ORIGINAL_PUBLICATION_JOB_ID="${PUBLICATION_JOB_ID}"
ROOT_FAILED_PREPARE_ARRAY_JOB_ID="${ROOT_FAILED_PREPARE_ARRAY_JOB_ID:-${ORIGINAL_PREPARE_ARRAY_JOB_ID}}"
ORIGINAL_JOB_LIST="${ORIGINAL_FREEZE_JOB_ID},${ORIGINAL_COHORT_JOB_ID},${ORIGINAL_PREPARE_ARRAY_JOB_ID},${ORIGINAL_EMBED_ARRAY_JOB_ID},${ORIGINAL_FIT_JOB_ID},${ORIGINAL_PUBLICATION_JOB_ID}"
if squeue -h -j "${ORIGINAL_JOB_LIST}" | grep -q .; then
  echo "Refusing recovery while the recorded Prompt 5 graph is still active or pending." >&2
  squeue -j "${ORIGINAL_JOB_LIST}" \
    -o "%.18i %.12P %.30j %.2t %.10M %.10l %R" >&2
  exit 1
fi

CURRENT_COMMIT="$(git rev-parse HEAD)"
MANIFEST_COMMIT="$(
  sed -n 's/^[[:space:]]*"source_git_commit": "\([0-9a-f]\{40\}\)",\{0,1\}$/\1/p' "${MANIFEST}" |
    head -n 1
)"
[[ "${MANIFEST_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Refusing recovery: frozen manifest source commit is unreadable." >&2
  exit 1
}

RECOVERY_CONFIG_SHA256=""
FAILED_PREPARE_SENTINEL_HASHES=()
if [[ "${CURRENT_COMMIT}" != "${MANIFEST_COMMIT}" ]]; then
  [[ -s "${RECOVERY_CONFIG}" ]] || {
    echo "Refusing cross-commit recovery without ${RECOVERY_CONFIG}." >&2
    exit 1
  }
  AUTHORIZED_FROZEN_COMMIT="$(
    sed -n 's/^frozen_source_git_commit:[[:space:]]*\([0-9a-f]\{40\}\)$/\1/p' "${RECOVERY_CONFIG}" |
      head -n 1
  )"
  AUTHORIZED_FAILED_PREPARE_ID="$(
    sed -n 's/^failed_prepare_array_job_id:[[:space:]]*\([0-9][0-9]*\)$/\1/p' "${RECOVERY_CONFIG}" |
      head -n 1
  )"
  if [[ "${MANIFEST_COMMIT}" != "${AUTHORIZED_FROZEN_COMMIT}" || \
        "${ROOT_FAILED_PREPARE_ARRAY_JOB_ID}" != "${AUTHORIZED_FAILED_PREPARE_ID}" ]]; then
    echo "Refusing recovery: the frozen manifest or failed preparation graph is not authorized." >&2
    exit 1
  fi
  for index in 0 1 2 3; do
    source_status_id="${ROOT_FAILED_PREPARE_ARRAY_JOB_ID}_${index}"
    failed_sentinel="outputs/private/run_status/${source_status_id}/FAILED.json"
    source_log="slurm-tehr-m1-test-prepare-${source_status_id}.out"
    [[ -s "${failed_sentinel}" && ! -e "outputs/private/run_status/${source_status_id}/COMPLETE.json" ]] || {
      echo "Refusing recovery: ${source_status_id} lacks an exclusive FAILED sentinel." >&2
      exit 1
    }
    grep -q "\"git_commit\": \"${MANIFEST_COMMIT}\"" "${failed_sentinel}" || {
      echo "Refusing recovery: ${source_status_id} did not fail at the frozen source commit." >&2
      exit 1
    }
    grep -q "M1 preparation prohibits the test split" "${source_log}" || {
      echo "Refusing recovery: ${source_status_id} does not show the authorized preparation-boundary defect." >&2
      exit 1
    }
    FAILED_PREPARE_SENTINEL_HASHES+=("$(sha256sum "${failed_sentinel}" | awk '{print $1}')")
  done
  RECOVERY_CONFIG_SHA256="$(sha256sum "${RECOVERY_CONFIG}" | awk '{print $1}')"
  export FINAL_M1_FROZEN_SOURCE_GIT_COMMIT="${MANIFEST_COMMIT}"
  export FINAL_STATUS_SOURCE_JOB_ID="${ROOT_FAILED_PREPARE_ARRAY_JOB_ID}"
  export FINAL_STATUS_SOURCE_GIT_COMMIT="${MANIFEST_COMMIT}"
  export FINAL_STATUS_RECOVERY_GIT_COMMIT="${CURRENT_COMMIT}"
fi

container_review() {
  srun \
    --partition=batch \
    --nodelist=dgit00038673 \
    --job-name=tehr-m1-resume-check \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:20:00 \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/data:/data \
    --container-workdir="${PROJECT_ROOT}" \
    --export=ALL \
    env PYTHONPATH=src FINAL_M1_GIT_COMMIT="${MANIFEST_COMMIT}" \
    .venv/bin/python scripts/review_final_submission.py --config "${CONFIG}" "$@"
}

submit_job() {
  local result
  result="$(sbatch --parsable "$@")"
  result="${result%%;*}"
  [[ "${result}" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: ${result}" >&2; exit 1; }
  printf '%s' "${result}"
}

container_review --require-frozen

if [[ -s "${PUBLICATION}" ]]; then
  container_review --require-publication
  echo "Prompt 5 publication artifacts are already COMPLETE; no recovery was submitted."
  exit 0
fi

if [[ -s "${RESULT}" ]]; then
  container_review --require-results
  PUBLICATION_JOB_ID="$(submit_job --export=ALL scripts/slurm/34_build_m1_publication_artifacts.sbatch)"
  FREEZE_JOB_ID="${ORIGINAL_FREEZE_JOB_ID}"
  COHORT_JOB_ID="${ORIGINAL_COHORT_JOB_ID}"
  PREPARE_ARRAY_JOB_ID="${ORIGINAL_PREPARE_ARRAY_JOB_ID}"
  EMBED_ARRAY_JOB_ID="${ORIGINAL_EMBED_ARRAY_JOB_ID}"
  FIT_JOB_ID="${ORIGINAL_FIT_JOB_ID}"
else
  if container_review --require-test-cohort >/dev/null 2>&1; then
    COHORT_JOB_ID="${ORIGINAL_COHORT_JOB_ID}"
    PREPARE_ARRAY_JOB_ID="$(
      submit_job --array=0-3%1 --export=ALL \
        scripts/slurm/31_prepare_m1_test_task_array.sbatch
    )"
  else
    COHORT_JOB_ID="$(submit_job --export=ALL scripts/slurm/30_build_m1_test_cohort.sbatch)"
    PREPARE_ARRAY_JOB_ID="$(
      submit_job --dependency="afterok:${COHORT_JOB_ID}" --array=0-3%1 --export=ALL \
        scripts/slurm/31_prepare_m1_test_task_array.sbatch
    )"
  fi
  EMBED_ARRAY_JOB_ID="$(submit_job --dependency="afterok:${PREPARE_ARRAY_JOB_ID}" --array=0-3%1 --export=ALL scripts/slurm/32_embed_m1_test_task_array.sbatch)"
  FIT_JOB_ID="$(submit_job --dependency="afterok:${EMBED_ARRAY_JOB_ID}" --export=ALL scripts/slurm/33_fit_m1_test_report.sbatch)"
  PUBLICATION_JOB_ID="$(submit_job --dependency="afterok:${FIT_JOB_ID}" --export=ALL scripts/slurm/34_build_m1_publication_artifacts.sbatch)"
  FREEZE_JOB_ID="${ORIGINAL_FREEZE_JOB_ID}"
fi

mkdir -p outputs/private/submissions
RECOVERY_FILE="outputs/private/submissions/prompt5-m1-same-manifest-recovery-${PUBLICATION_JOB_ID}.env"
[[ ! -e "${RECOVERY_FILE}" ]] || { echo "Refusing to overwrite ${RECOVERY_FILE}" >&2; exit 1; }
TEMPORARY_FILE="$(mktemp "${RECOVERY_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${MANIFEST_COMMIT}"
  printf 'RECOVERY_GIT_COMMIT=%q\n' "${CURRENT_COMMIT}"
  printf 'RECOVERY_CONFIG=%q\n' "${RECOVERY_CONFIG}"
  printf 'RECOVERY_CONFIG_SHA256=%q\n' "${RECOVERY_CONFIG_SHA256}"
  printf 'ROOT_FAILED_PREPARE_ARRAY_JOB_ID=%q\n' "${ROOT_FAILED_PREPARE_ARRAY_JOB_ID}"
  for index in 0 1 2 3; do
    printf 'ROOT_FAILED_PREPARE_SENTINEL_%s_SHA256=%q\n' \
      "${index}" "${FAILED_PREPARE_SENTINEL_HASHES[${index}]:-}"
  done
  printf 'FREEZE_JOB_ID=%q\n' "${FREEZE_JOB_ID}"
  printf 'COHORT_JOB_ID=%q\n' "${COHORT_JOB_ID}"
  printf 'PREPARE_ARRAY_JOB_ID=%q\n' "${PREPARE_ARRAY_JOB_ID}"
  printf 'EMBED_ARRAY_JOB_ID=%q\n' "${EMBED_ARRAY_JOB_ID}"
  printf 'FIT_JOB_ID=%q\n' "${FIT_JOB_ID}"
  printf 'PUBLICATION_JOB_ID=%q\n' "${PUBLICATION_JOB_ID}"
  printf 'RECOVERY_OF_PREPARE_ARRAY_JOB_ID=%q\n' "${ORIGINAL_PREPARE_ARRAY_JOB_ID}"
  printf 'RECOVERY_OF_EMBED_ARRAY_JOB_ID=%q\n' "${ORIGINAL_EMBED_ARRAY_JOB_ID}"
  printf 'RECOVERY_OF_FIT_JOB_ID=%q\n' "${ORIGINAL_FIT_JOB_ID}"
  printf 'RECOVERY_OF_PUBLICATION_JOB_ID=%q\n' "${ORIGINAL_PUBLICATION_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${RECOVERY_FILE}"
trap - EXIT
ln -sfn "submissions/$(basename "${RECOVERY_FILE}")" outputs/private/prompt5_job_ids.env
cat "${RECOVERY_FILE}"
