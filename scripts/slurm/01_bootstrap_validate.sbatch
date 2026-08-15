#!/usr/bin/env bash
#SBATCH --job-name=tehr-bootstrap
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:-/data/shared73/projects/wdelery/model_cache/qwen3-embedding}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before sbatch --export=ALL}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before sbatch --export=ALL}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"

mkdir -p "${PROJECT_ROOT}/logs/slurm" "${MODEL_ROOT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc '
    set -euo pipefail
    python3 -m venv --system-site-packages .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -e ".[dev,model]"
    export PYTHONPATH=src
    .venv/bin/python scripts/dgx/check_environment.py \
      --project-root "${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}" \
      --model-root "${TEMPORAL_EHR_MODEL_ROOT:-/data/shared73/projects/wdelery/model_cache/qwen3-embedding}"
    .venv/bin/python -m temporal_ehr.cli validate-labels \
      --asset-root "${EHRSHOT_ASSET_ROOT}" \
      --output-dir reports/generated/label_validation
    .venv/bin/python -m pytest
    .venv/bin/python -m ruff check .
  '
