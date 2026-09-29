#!/usr/bin/env bash
set -euo pipefail
export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
cd "$TEMPORAL_EHR_PROJECT_ROOT"
source outputs/private/m2_production_fit_job_ids.env
export M2_PRODUCTION_COMMIT
srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=32G --time=01:00:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/run_m2_structured_production.py verify \
  --array "$M2_PRODUCTION_EMBED_ARRAY" --fit-job "$M2_PRODUCTION_FIT_JOB"
