#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
STAGE="${1:-reference}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"
M2_SUMMARY_PARALLELISM="${PROMPT4_M2_SUMMARY_PARALLELISM:-4}"
M2_EMBED_PARALLELISM="${PROMPT4_M2_EMBED_PARALLELISM:-1}"
M3_PREPARE_PARALLELISM="${PROMPT4_M3_PREPARE_PARALLELISM:-1}"
M3_TRAIN_PARALLELISM="${PROMPT4_M3_TRAIN_PARALLELISM:-2}"

require_parallelism() {
  local name="$1" value="$2" maximum="$3"
  if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]] || (( value > maximum )); then
    echo "${name} must be an integer from 1 through ${maximum}." >&2
    exit 2
  fi
}

require_parallelism PROMPT4_M2_SUMMARY_PARALLELISM "${M2_SUMMARY_PARALLELISM}" 4
require_parallelism PROMPT4_M2_EMBED_PARALLELISM "${M2_EMBED_PARALLELISM}" 2
require_parallelism PROMPT4_M3_PREPARE_PARALLELISM "${M3_PREPARE_PARALLELISM}" 2
require_parallelism PROMPT4_M3_TRAIN_PARALLELISM "${M3_TRAIN_PARALLELISM}" 3

if [[ "${STAGE}" != "reference" && "${STAGE}" != "reference-finalize" && "${STAGE}" != "sensitivity" ]]; then
  echo "Usage: bash scripts/slurm/submit_prompt4_execution.sh [reference|reference-finalize|sensitivity]" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Prompt 4 submission requires a clean DGX checkout." >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
export TEMPORAL_EHR_PROJECT_ROOT="${PROJECT_ROOT}"
export TEMPORAL_EHR_MODEL_ROOT="${MODEL_ROOT}"
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export FINAL_M23_GIT_COMMIT="${SOURCE_GIT_COMMIT}"

if [[ "${STAGE}" == "reference" ]]; then
  echo "Direct reference submission is superseded; run submit_prompt4_m2_fidelity_v7.sh." >&2
  exit 1
fi

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
  env PYTHONPATH=src "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates

mkdir -p outputs/private/submissions

if [[ "${STAGE}" == "reference" ]]; then
  M2_SUMMARY_SHARDS_PER_TASK=192
  M2_REFERENCE_MAX_INDEX=$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))
  M2_ARRAY_JOB_ID="$(
    sbatch --parsable --array="0-${M2_REFERENCE_MAX_INDEX}%${M2_SUMMARY_PARALLELISM}" \
      --export=ALL,M23_STAGE=reference,M2_SUMMARY_SHARDS_PER_TASK="${M2_SUMMARY_SHARDS_PER_TASK}" \
      scripts/slurm/24c_m2_summary_shard_array.sbatch
  )"
  M2_EMBED_ARRAY_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${M2_ARRAY_JOB_ID}" \
      --array="0-3%${M2_EMBED_PARALLELISM}" \
      --export=ALL,M23_STAGE=reference scripts/slurm/24b_m2_embed_task_array.sbatch
  )"
  M2_REVIEW_PREP_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${M2_ARRAY_JOB_ID}" \
      --export=ALL scripts/slurm/25a_prepare_m2_fidelity_review.sbatch
  )"
  M3_PREP_ARRAY_JOB_ID="$(
    sbatch --parsable --array="0-3%${M3_PREPARE_PARALLELISM}" \
      --export=ALL,M23_STAGE=reference scripts/slurm/26a_m3_prepare_task_array.sbatch
  )"
  M3_ARRAY_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${M3_PREP_ARRAY_JOB_ID}" \
      --array="0-11%${M3_TRAIN_PARALLELISM}" \
      --export=ALL,M23_STAGE=reference scripts/slurm/26_m3_task_seed_array.sbatch
  )"
  SUBMISSION_FILE="outputs/private/submissions/prompt4-reference-${M2_ARRAY_JOB_ID}.env"
  {
    printf 'REFERENCE_SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
    printf 'PROMPT4_STAGE=%q\n' "reference"
    printf 'M2_ARRAY_JOB_ID=%q\n' "${M2_ARRAY_JOB_ID}"
    printf 'M2_SUMMARY_SHARDS_PER_TASK=%q\n' "${M2_SUMMARY_SHARDS_PER_TASK}"
    printf 'M2_EMBED_ARRAY_JOB_ID=%q\n' "${M2_EMBED_ARRAY_JOB_ID}"
    printf 'M2_REVIEW_PREP_JOB_ID=%q\n' "${M2_REVIEW_PREP_JOB_ID}"
    printf 'M3_PREP_ARRAY_JOB_ID=%q\n' "${M3_PREP_ARRAY_JOB_ID}"
    printf 'M3_ARRAY_JOB_ID=%q\n' "${M3_ARRAY_JOB_ID}"
    printf 'M2_SUMMARY_PARALLELISM=%q\n' "${M2_SUMMARY_PARALLELISM}"
    printf 'M2_EMBED_PARALLELISM=%q\n' "${M2_EMBED_PARALLELISM}"
    printf 'M3_PREPARE_PARALLELISM=%q\n' "${M3_PREPARE_PARALLELISM}"
    printf 'M3_TRAIN_PARALLELISM=%q\n' "${M3_TRAIN_PARALLELISM}"
  } > "${SUBMISSION_FILE}"
  ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" outputs/private/prompt4_reference_job_ids.env
  printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
  printf 'M2_ARRAY_JOB_ID=%s\n' "${M2_ARRAY_JOB_ID}"
  printf 'M2_EMBED_ARRAY_JOB_ID=%s\n' "${M2_EMBED_ARRAY_JOB_ID}"
  printf 'M2_REVIEW_PREP_JOB_ID=%s\n' "${M2_REVIEW_PREP_JOB_ID}"
  printf 'M3_PREP_ARRAY_JOB_ID=%s\n' "${M3_PREP_ARRAY_JOB_ID}"
  printf 'M3_ARRAY_JOB_ID=%s\n' "${M3_ARRAY_JOB_ID}"
  printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
  exit 0
fi

source outputs/private/prompt4_reference_job_ids.env
if [[ "${REFERENCE_SOURCE_GIT_COMMIT}" != "${SOURCE_GIT_COMMIT}" ]]; then
  echo "Prompt 4 reference and finalize/sensitivity source commits must match." >&2
  exit 1
fi
if [[ "${STAGE}" == "reference-finalize" ]]; then
  export PROMPT4_M3_EVIDENCE_SOURCE_GIT_COMMIT="${M3_EVIDENCE_SOURCE_GIT_COMMIT:-${SOURCE_GIT_COMMIT}}"
  export PROMPT4_M2_RECOVERY_REPORT="${M2_RECOVERY_REPORT:-}"
  export PROMPT4_M2_RECOVERY_REPORT_SHA256="${M2_RECOVERY_REPORT_SHA256:-}"
else
  export PROMPT4_M3_EVIDENCE_SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}"
fi

if [[ "${STAGE}" == "reference-finalize" ]]; then
  STATUS_IDS=("${M2_REVIEW_PREP_JOB_ID}")
  M2_REFERENCE_MAX_INDEX=3
  if [[ -n "${M2_SUMMARY_SHARDS_PER_TASK:-}" ]]; then
    M2_REFERENCE_MAX_INDEX=$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))
  fi
  for index in $(seq 0 "${M2_REFERENCE_MAX_INDEX}"); do
    STATUS_IDS+=("${M2_ARRAY_JOB_ID}_${index}")
  done
  for index in $(seq 0 3); do STATUS_IDS+=("${M2_EMBED_ARRAY_JOB_ID}_${index}"); done
  for index in $(seq 0 3); do STATUS_IDS+=("${M3_PREP_ARRAY_JOB_ID}_${index}"); done
  for index in $(seq 0 11); do STATUS_IDS+=("${M3_ARRAY_JOB_ID}_${index}"); done
  for status_id in "${STATUS_IDS[@]}"; do
    if [[ ! -s "outputs/private/run_status/${status_id}/COMPLETE.json" || \
          -e "outputs/private/run_status/${status_id}/FAILED.json" ]]; then
      echo "Prompt 4 reference compute is not COMPLETE for ${status_id}." >&2
      exit 1
    fi
  done
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
    env PYTHONPATH=src "${PYTHON}" scripts/review_m2_fidelity_manual.py finalize --config "${CONFIG}"
  M2_FIT_JOB_ID="$(sbatch --parsable --export=ALL,M23_STAGE=reference scripts/slurm/25_m2_fit_validation.sbatch)"
  M3_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${M2_FIT_JOB_ID}" \
      --export=ALL,M23_STAGE=reference scripts/slurm/27_m3_fit_validation.sbatch
  )"
  SUBMISSION_FILE="outputs/private/submissions/prompt4-reference-finalize-${M2_FIT_JOB_ID}.env"
  {
    printf 'REFERENCE_SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
    printf 'PROMPT4_STAGE=%q\n' "reference-finalize"
    printf 'M2_FIT_JOB_ID=%q\n' "${M2_FIT_JOB_ID}"
    printf 'M3_FIT_JOB_ID=%q\n' "${M3_FIT_JOB_ID}"
  } > "${SUBMISSION_FILE}"
  ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" \
    outputs/private/prompt4_reference_finalize_job_ids.env
  printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
  printf 'M2_FIT_JOB_ID=%s\n' "${M2_FIT_JOB_ID}"
  printf 'M3_FIT_JOB_ID=%s\n' "${M3_FIT_JOB_ID}"
  printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
  exit 0
fi

SELECTION_PATH="reports/generated/final/locked_four_task_m2_m3_v1/approach_selection.json"
if [[ ! -s "${SELECTION_PATH}" ]]; then
  echo "Prompt 4 sensitivity requires a completed reference selection." >&2
  exit 1
fi
SELECTED_APPROACH="$(
  SELECTION_PATH="${SELECTION_PATH}" "${PYTHON}" -c \
    'import json, os; print(json.load(open(os.environ["SELECTION_PATH"]))["selected_approach"])'
)"
source outputs/private/prompt4_reference_finalize_job_ids.env
M2_REFERENCE_MAX_INDEX=3
if [[ -n "${M2_SUMMARY_SHARDS_PER_TASK:-}" ]]; then
  M2_REFERENCE_MAX_INDEX=$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))
fi
REFERENCE_STATUS_IDS=""
for index in $(seq 0 "${M2_REFERENCE_MAX_INDEX}"); do
  [[ -n "${REFERENCE_STATUS_IDS}" ]] && REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},"
  REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS}${M2_ARRAY_JOB_ID}_${index}"
done
REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M2_REVIEW_PREP_JOB_ID},${M2_FIT_JOB_ID}"
REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M2_EMBED_ARRAY_JOB_ID}_0,${M2_EMBED_ARRAY_JOB_ID}_1,${M2_EMBED_ARRAY_JOB_ID}_2,${M2_EMBED_ARRAY_JOB_ID}_3"
REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M3_PREP_ARRAY_JOB_ID}_0,${M3_PREP_ARRAY_JOB_ID}_1,${M3_PREP_ARRAY_JOB_ID}_2,${M3_PREP_ARRAY_JOB_ID}_3"
for index in $(seq 0 11); do
  REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M3_ARRAY_JOB_ID}_${index}"
done
REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M3_FIT_JOB_ID}"

if [[ "${SELECTED_APPROACH}" == "M2" ]]; then
  M2_SUMMARY_SHARDS_PER_TASK="${M2_SUMMARY_SHARDS_PER_TASK:-192}"
  M2_SENSITIVITY_MAX_INDEX=$((4 * M2_SUMMARY_SHARDS_PER_TASK - 1))
  SENSITIVITY_SUMMARY_ARRAY_JOB_ID="$(
    sbatch --parsable --array="0-${M2_SENSITIVITY_MAX_INDEX}%${M2_SUMMARY_PARALLELISM}" \
      --export=ALL,M23_STAGE=sensitivity,M2_SUMMARY_SHARDS_PER_TASK="${M2_SUMMARY_SHARDS_PER_TASK}" \
      scripts/slurm/24c_m2_summary_shard_array.sbatch
  )"
  SENSITIVITY_EMBED_ARRAY_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}" \
      --array="0-3%${M2_EMBED_PARALLELISM}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/24b_m2_embed_task_array.sbatch
  )"
  SENSITIVITY_ARRAY_JOB_ID="${SENSITIVITY_EMBED_ARRAY_JOB_ID}"
  SENSITIVITY_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_EMBED_ARRAY_JOB_ID}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/25_m2_fit_validation.sbatch
  )"
  SENSITIVITY_STATUS_IDS=""
  for index in $(seq 0 "${M2_SENSITIVITY_MAX_INDEX}"); do
    [[ -n "${SENSITIVITY_STATUS_IDS}" ]] && SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS},"
    SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS}${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}_${index}"
  done
  SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS},${SENSITIVITY_EMBED_ARRAY_JOB_ID}_0,${SENSITIVITY_EMBED_ARRAY_JOB_ID}_1,${SENSITIVITY_EMBED_ARRAY_JOB_ID}_2,${SENSITIVITY_EMBED_ARRAY_JOB_ID}_3,${SENSITIVITY_FIT_JOB_ID}"
  SENSITIVITY_M3_PREP_ARRAY_JOB_ID=""
elif [[ "${SELECTED_APPROACH}" == "M3" ]]; then
  SENSITIVITY_SUMMARY_ARRAY_JOB_ID=""
  SENSITIVITY_EMBED_ARRAY_JOB_ID=""
  SENSITIVITY_M3_PREP_ARRAY_JOB_ID="$(
    sbatch --parsable --array="0-3%${M3_PREPARE_PARALLELISM}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/26a_m3_prepare_task_array.sbatch
  )"
  SENSITIVITY_ARRAY_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}" \
      --array="0-11%${M3_TRAIN_PARALLELISM}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/26_m3_task_seed_array.sbatch
  )"
  SENSITIVITY_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_ARRAY_JOB_ID}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/27_m3_fit_validation.sbatch
  )"
  SENSITIVITY_STATUS_IDS="${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}_0,${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}_1,${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}_2,${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}_3"
  for index in $(seq 0 11); do
    SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS},"
    SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS}${SENSITIVITY_ARRAY_JOB_ID}_${index}"
  done
  SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS},${SENSITIVITY_FIT_JOB_ID}"
elif [[ "${SELECTED_APPROACH}" == "M1" ]]; then
  SENSITIVITY_SUMMARY_ARRAY_JOB_ID=""
  SENSITIVITY_EMBED_ARRAY_JOB_ID=""
  SENSITIVITY_ARRAY_JOB_ID=""
  SENSITIVITY_FIT_JOB_ID=""
  SENSITIVITY_M3_PREP_ARRAY_JOB_ID=""
  SENSITIVITY_STATUS_IDS=""
else
  echo "Unexpected selected approach: ${SELECTED_APPROACH}" >&2
  exit 1
fi

ALL_STATUS_IDS="${REFERENCE_STATUS_IDS}"
[[ -n "${SENSITIVITY_STATUS_IDS}" ]] && ALL_STATUS_IDS="${ALL_STATUS_IDS},${SENSITIVITY_STATUS_IDS}"
if [[ -n "${SENSITIVITY_FIT_JOB_ID}" ]]; then
  FREEZE_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_FIT_JOB_ID}" \
      --export=ALL,PROMPT4_STATUS_IDS="${ALL_STATUS_IDS}" scripts/slurm/28_freeze_protocol.sbatch
  )"
else
  FREEZE_JOB_ID="$(
    sbatch --parsable --export=ALL,PROMPT4_STATUS_IDS="${ALL_STATUS_IDS}" \
      scripts/slurm/28_freeze_protocol.sbatch
  )"
fi
SUBMISSION_FILE="outputs/private/submissions/prompt4-sensitivity-${FREEZE_JOB_ID}.env"
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'PROMPT4_STAGE=%q\n' "sensitivity"
  printf 'SELECTED_APPROACH=%q\n' "${SELECTED_APPROACH}"
  printf 'SENSITIVITY_SUMMARY_ARRAY_JOB_ID=%q\n' "${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}"
  printf 'M2_SUMMARY_SHARDS_PER_TASK=%q\n' "${M2_SUMMARY_SHARDS_PER_TASK:-}"
  printf 'SENSITIVITY_EMBED_ARRAY_JOB_ID=%q\n' "${SENSITIVITY_EMBED_ARRAY_JOB_ID}"
  printf 'SENSITIVITY_M3_PREP_ARRAY_JOB_ID=%q\n' "${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}"
  printf 'SENSITIVITY_ARRAY_JOB_ID=%q\n' "${SENSITIVITY_ARRAY_JOB_ID}"
  printf 'SENSITIVITY_FIT_JOB_ID=%q\n' "${SENSITIVITY_FIT_JOB_ID}"
  printf 'FREEZE_JOB_ID=%q\n' "${FREEZE_JOB_ID}"
} > "${SUBMISSION_FILE}"
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" outputs/private/prompt4_sensitivity_job_ids.env
printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'SELECTED_APPROACH=%s\n' "${SELECTED_APPROACH}"
printf 'SENSITIVITY_SUMMARY_ARRAY_JOB_ID=%s\n' "${SENSITIVITY_SUMMARY_ARRAY_JOB_ID}"
printf 'SENSITIVITY_EMBED_ARRAY_JOB_ID=%s\n' "${SENSITIVITY_EMBED_ARRAY_JOB_ID}"
printf 'SENSITIVITY_M3_PREP_ARRAY_JOB_ID=%s\n' "${SENSITIVITY_M3_PREP_ARRAY_JOB_ID}"
printf 'SENSITIVITY_ARRAY_JOB_ID=%s\n' "${SENSITIVITY_ARRAY_JOB_ID}"
printf 'SENSITIVITY_FIT_JOB_ID=%s\n' "${SENSITIVITY_FIT_JOB_ID}"
printf 'FREEZE_JOB_ID=%s\n' "${FREEZE_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
