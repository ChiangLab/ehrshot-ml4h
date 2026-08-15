#!/usr/bin/env bash
#SBATCH --job-name=tehr-censor-audit
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before sbatch --export=ALL}"
COHORT_PATH="${EHRSHOT_COHORT_PATH:-${PROJECT_ROOT}/outputs/private/cohorts/primary_panel.parquet}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  env PYTHONPATH=src .venv/bin/python -m temporal_ehr.cli validate-censoring \
    --data-path "${EVENT_PATH}" \
    --cohort-path "${COHORT_PATH}" \
    --output-dir reports/generated/censoring
