#!/usr/bin/env bash
# Read-only semantic verification after the manual gate and fit complete.
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_selection_v4_job_ids.env
source outputs/private/m2_selection_v4_fit_job_ids.env
export TEMPORAL_EHR_PROJECT_ROOT="$PWD"
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
export M2_SELECTION_COMMIT
test "$(git rev-parse HEAD)" = "$M2_SELECTION_COMMIT"
test -z "$(git status --porcelain)"
srun \
  --partition=batch \
  --nodelist=dgit00038673 \
  --cpus-per-task=2 \
  --mem=16G \
  --time=01:00:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data \
  --container-workdir="$PWD" \
  --export=ALL \
  env PYTHONPATH=src \
  .venv/bin/python scripts/run_m2_selection_v4.py verify \
    --array "$M2_SELECTION_EMBED_ARRAY" \
    --fit-job "$M2_SELECTION_FIT_JOB"
echo 'PASS: selection-v4 M2 reference result recursively verified; test remained closed.'
