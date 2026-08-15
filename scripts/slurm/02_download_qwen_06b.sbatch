#!/usr/bin/env bash
#SBATCH --job-name=qwen3e-06b-download
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:-/data/shared73/projects/wdelery/model_cache/qwen3-embedding}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
mkdir -p "${PROJECT_ROOT}/logs/slurm" "${MODEL_ROOT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  .venv/bin/python scripts/dgx/download_qwen_embeddings.py \
    --models 0.6B \
    --output-root "${MODEL_ROOT}"
