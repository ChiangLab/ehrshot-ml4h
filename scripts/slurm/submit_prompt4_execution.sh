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

if [[ "${STAGE}" != "reference" && "${STAGE}" != "sensitivity" ]]; then
  echo "Usage: bash scripts/slurm/submit_prompt4_execution.sh [reference|sensitivity]" >&2
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
  M2_ARRAY_JOB_ID="$(sbatch --parsable --export=ALL,M23_STAGE=reference scripts/slurm/24_m2_summary_task_array.sbatch)"
  M2_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${M2_ARRAY_JOB_ID}" \
      --export=ALL,M23_STAGE=reference scripts/slurm/25_m2_fit_validation.sbatch
  )"
  M3_ARRAY_JOB_ID="$(sbatch --parsable --export=ALL,M23_STAGE=reference scripts/slurm/26_m3_task_seed_array.sbatch)"
  M3_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${M2_FIT_JOB_ID}:${M3_ARRAY_JOB_ID}" \
      --export=ALL,M23_STAGE=reference scripts/slurm/27_m3_fit_validation.sbatch
  )"
  SUBMISSION_FILE="outputs/private/submissions/prompt4-reference-${M2_ARRAY_JOB_ID}.env"
  {
    printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
    printf 'PROMPT4_STAGE=%q\n' "reference"
    printf 'M2_ARRAY_JOB_ID=%q\n' "${M2_ARRAY_JOB_ID}"
    printf 'M2_FIT_JOB_ID=%q\n' "${M2_FIT_JOB_ID}"
    printf 'M3_ARRAY_JOB_ID=%q\n' "${M3_ARRAY_JOB_ID}"
    printf 'M3_FIT_JOB_ID=%q\n' "${M3_FIT_JOB_ID}"
  } > "${SUBMISSION_FILE}"
  ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" outputs/private/prompt4_reference_job_ids.env
  printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
  printf 'M2_ARRAY_JOB_ID=%s\n' "${M2_ARRAY_JOB_ID}"
  printf 'M2_FIT_JOB_ID=%s\n' "${M2_FIT_JOB_ID}"
  printf 'M3_ARRAY_JOB_ID=%s\n' "${M3_ARRAY_JOB_ID}"
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
source outputs/private/prompt4_reference_job_ids.env
REFERENCE_STATUS_IDS="${M2_ARRAY_JOB_ID}_0,${M2_ARRAY_JOB_ID}_1,${M2_ARRAY_JOB_ID}_2,${M2_ARRAY_JOB_ID}_3,${M2_FIT_JOB_ID}"
for index in $(seq 0 11); do
  REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M3_ARRAY_JOB_ID}_${index}"
done
REFERENCE_STATUS_IDS="${REFERENCE_STATUS_IDS},${M3_FIT_JOB_ID}"

if [[ "${SELECTED_APPROACH}" == "M2" ]]; then
  SENSITIVITY_ARRAY_JOB_ID="$(
    sbatch --parsable --export=ALL,M23_STAGE=sensitivity scripts/slurm/24_m2_summary_task_array.sbatch
  )"
  SENSITIVITY_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_ARRAY_JOB_ID}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/25_m2_fit_validation.sbatch
  )"
  SENSITIVITY_STATUS_IDS="${SENSITIVITY_ARRAY_JOB_ID}_0,${SENSITIVITY_ARRAY_JOB_ID}_1,${SENSITIVITY_ARRAY_JOB_ID}_2,${SENSITIVITY_ARRAY_JOB_ID}_3,${SENSITIVITY_FIT_JOB_ID}"
elif [[ "${SELECTED_APPROACH}" == "M3" ]]; then
  SENSITIVITY_ARRAY_JOB_ID="$(
    sbatch --parsable --export=ALL,M23_STAGE=sensitivity scripts/slurm/26_m3_task_seed_array.sbatch
  )"
  SENSITIVITY_FIT_JOB_ID="$(
    sbatch --parsable --dependency="afterok:${SENSITIVITY_ARRAY_JOB_ID}" \
      --export=ALL,M23_STAGE=sensitivity scripts/slurm/27_m3_fit_validation.sbatch
  )"
  SENSITIVITY_STATUS_IDS=""
  for index in $(seq 0 11); do
    [[ -n "${SENSITIVITY_STATUS_IDS}" ]] && SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS},"
    SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS}${SENSITIVITY_ARRAY_JOB_ID}_${index}"
  done
  SENSITIVITY_STATUS_IDS="${SENSITIVITY_STATUS_IDS},${SENSITIVITY_FIT_JOB_ID}"
elif [[ "${SELECTED_APPROACH}" == "M1" ]]; then
  SENSITIVITY_ARRAY_JOB_ID=""
  SENSITIVITY_FIT_JOB_ID=""
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
  printf 'SENSITIVITY_ARRAY_JOB_ID=%q\n' "${SENSITIVITY_ARRAY_JOB_ID}"
  printf 'SENSITIVITY_FIT_JOB_ID=%q\n' "${SENSITIVITY_FIT_JOB_ID}"
  printf 'FREEZE_JOB_ID=%q\n' "${FREEZE_JOB_ID}"
} > "${SUBMISSION_FILE}"
ln -sfn "submissions/$(basename "${SUBMISSION_FILE}")" outputs/private/prompt4_sensitivity_job_ids.env
printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'SELECTED_APPROACH=%s\n' "${SELECTED_APPROACH}"
printf 'SENSITIVITY_ARRAY_JOB_ID=%s\n' "${SENSITIVITY_ARRAY_JOB_ID}"
printf 'SENSITIVITY_FIT_JOB_ID=%s\n' "${SENSITIVITY_FIT_JOB_ID}"
printf 'FREEZE_JOB_ID=%s\n' "${FREEZE_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
