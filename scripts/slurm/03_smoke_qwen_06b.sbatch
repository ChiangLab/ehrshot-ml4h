#!/usr/bin/env bash
#SBATCH --job-name=qwen3e-06b-smoke
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --gres=mps:1

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
MODEL_ROOT="${TEMPORAL_EHR_MODEL_ROOT:-/data/shared73/projects/wdelery/model_cache/qwen3-embedding}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
mkdir -p "${PROJECT_ROOT}/logs/slurm"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  env PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv/bin/python scripts/dgx/smoke_test_qwen_embeddings.py \
    --model-path "${MODEL_ROOT}/Qwen3-Embedding-0.6B" \
    --output "outputs/private/smoke/qwen_embedding_smoke-${SLURM_JOB_ID}.json" \
    --max-seq-length 512
