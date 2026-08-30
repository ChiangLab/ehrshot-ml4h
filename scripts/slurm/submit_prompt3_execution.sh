#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:?Set TEMPORAL_EHR_MODEL_ROOT before submission}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
PRIMARY_MODEL="${MODEL_ROOT}/Qwen3-Embedding-0.6B"
CONFIRMATORY_MODEL="${MODEL_ROOT}/Qwen3-Embedding-8B"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
EXECUTION_NODE="${TEMPORAL_EHR_EXECUTION_NODE:-dgit00038673}"

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing Prompt 3 submission from a dirty Git worktree." >&2
  git status --short >&2
  exit 1
fi
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ ! "${SOURCE_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Prompt 3 requires a full source Git commit." >&2
  exit 1
fi
for path in \
  "${EVENT_PATH}" \
  "${PRIMARY_MODEL}/PINNED_REVISION.txt" \
  "${PRIMARY_MODEL}/MODEL_ARTIFACT_MANIFEST.json" \
  "${CONFIRMATORY_MODEL}/PINNED_REVISION.txt" \
  "${CONFIRMATORY_MODEL}/MODEL_ARTIFACT_MANIFEST.json" \
  configs/experiments/final_four_task_m1.yaml \
  configs/experiments/locked_panel.yaml \
  reports/generated/handoff/project_handoff.json \
  reports/generated/handoff/pilot_handoff.json \
  reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json; do
  if [[ ! -s "${path}" ]]; then
    echo "Missing required Prompt 3 input: ${path}" >&2
    if [[ "${path}" == "${CONFIRMATORY_MODEL}"/* ]]; then
      echo "Download and smoke-test the complete 8B snapshot before Prompt 3 submission." >&2
    fi
    exit 1
  fi
done
if [[ ! -d "${ASSET_ROOT}" ]]; then
  echo "Missing EHRSHOT asset directory: ${ASSET_ROOT}" >&2
  exit 1
fi

srun \
  --partition=batch \
  --nodelist="${EXECUTION_NODE}" \
  --job-name=tehr-prompt3-preflight \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:10:00 \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/run_final_m1.py hard-gates

mkdir -p outputs/private/submissions
SMOKE_8B_JOB_ID="$(
  sbatch --parsable --partition=batch --export=ALL scripts/slurm/06_smoke_qwen_8b.sbatch
)"
SMOKE_8B_JOB_ID="${SMOKE_8B_JOB_ID%%;*}"
FOUNDATION_JOB_ID="$(
  sbatch --parsable --partition=batch --export=ALL scripts/slurm/20_final_foundation.sbatch
)"
FOUNDATION_JOB_ID="${FOUNDATION_JOB_ID%%;*}"
THROUGHPUT_JOB_ID="$(
  sbatch --parsable --partition=batch \
    --dependency="afterok:${FOUNDATION_JOB_ID}:${SMOKE_8B_JOB_ID}" --export=ALL \
    scripts/slurm/21_final_throughput.sbatch
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
for job_id in \
  "${SMOKE_8B_JOB_ID}" \
  "${FOUNDATION_JOB_ID}" \
  "${THROUGHPUT_JOB_ID}" \
  "${EMBED_ARRAY_JOB_ID}" \
  "${FIT_JOB_ID}"; do
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Prompt 3 submission did not return one numeric job ID: ${job_id}" >&2
    exit 1
  fi
done

SUBMISSION_FILE="outputs/private/submissions/prompt3-${FOUNDATION_JOB_ID}.env"
if [[ -e "${SUBMISSION_FILE}" ]]; then
  echo "Refusing to overwrite existing submission record: ${SUBMISSION_FILE}" >&2
  exit 1
fi
TEMPORARY_FILE="$(mktemp "${SUBMISSION_FILE}.tmp.XXXXXX")"
trap 'rm -f "${TEMPORARY_FILE}"' EXIT
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'SMOKE_8B_JOB_ID=%q\n' "${SMOKE_8B_JOB_ID}"
  printf 'FOUNDATION_JOB_ID=%q\n' "${FOUNDATION_JOB_ID}"
  printf 'THROUGHPUT_JOB_ID=%q\n' "${THROUGHPUT_JOB_ID}"
  printf 'EMBED_ARRAY_JOB_ID=%q\n' "${EMBED_ARRAY_JOB_ID}"
  printf 'FIT_JOB_ID=%q\n' "${FIT_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${SUBMISSION_FILE}"
trap - EXIT
ln -sfn "submissions/prompt3-${FOUNDATION_JOB_ID}.env" outputs/private/prompt3_job_ids.env

printf 'SOURCE_GIT_COMMIT=%s\n' "${SOURCE_GIT_COMMIT}"
printf 'SMOKE_8B_JOB_ID=%s\n' "${SMOKE_8B_JOB_ID}"
printf 'FOUNDATION_JOB_ID=%s\n' "${FOUNDATION_JOB_ID}"
printf 'THROUGHPUT_JOB_ID=%s\n' "${THROUGHPUT_JOB_ID}"
printf 'EMBED_ARRAY_JOB_ID=%s\n' "${EMBED_ARRAY_JOB_ID}"
printf 'FIT_JOB_ID=%s\n' "${FIT_JOB_ID}"
printf 'SUBMISSION_FILE=%s\n' "${SUBMISSION_FILE}"
