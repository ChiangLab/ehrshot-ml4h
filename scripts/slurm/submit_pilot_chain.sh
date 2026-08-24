#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
cd "${PROJECT_ROOT}"

for variable in EHRSHOT_DATA_PATH EHRSHOT_ASSET_ROOT TEMPORAL_EHR_MODEL_ROOT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Set ${variable} before submitting the pilot chain." >&2
    exit 1
  fi
done

prepare_dependency=()
if [[ -n "${PILOT_AFTEROK_JOB_ID:-}" ]]; then
  if [[ ! "${PILOT_AFTEROK_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "PILOT_AFTEROK_JOB_ID must be one numeric Slurm job ID." >&2
    exit 1
  fi
  prepare_dependency+=(--dependency="afterok:${PILOT_AFTEROK_JOB_ID}")
fi

PREPARE_JOB_ID="$(
  sbatch --parsable --partition=batch "${prepare_dependency[@]}" --export=ALL \
    scripts/slurm/12_pilot_prepare.sbatch
)"
PREPARE_JOB_ID="${PREPARE_JOB_ID%%;*}"
THROUGHPUT_JOB_ID="$(
  sbatch --parsable --partition=batch --dependency="afterok:${PREPARE_JOB_ID}" --export=ALL \
    scripts/slurm/13_pilot_throughput.sbatch
)"
THROUGHPUT_JOB_ID="${THROUGHPUT_JOB_ID%%;*}"
EMBED_ARRAY_JOB_ID="$(
  sbatch --parsable --partition=batch --dependency="afterok:${THROUGHPUT_JOB_ID}" --export=ALL \
    scripts/slurm/14_pilot_embed_task_array.sbatch
)"
EMBED_ARRAY_JOB_ID="${EMBED_ARRAY_JOB_ID%%;*}"
REPORT_JOB_ID="$(
  sbatch --parsable --partition=batch --dependency="afterok:${EMBED_ARRAY_JOB_ID}" --export=ALL \
    scripts/slurm/15_pilot_fit_report.sbatch
)"
REPORT_JOB_ID="${REPORT_JOB_ID%%;*}"

printf 'PREPARE_JOB_ID=%s\n' "${PREPARE_JOB_ID}"
printf 'THROUGHPUT_JOB_ID=%s\n' "${THROUGHPUT_JOB_ID}"
printf 'EMBED_ARRAY_JOB_ID=%s\n' "${EMBED_ARRAY_JOB_ID}"
printf 'REPORT_JOB_ID=%s\n' "${REPORT_JOB_ID}"
