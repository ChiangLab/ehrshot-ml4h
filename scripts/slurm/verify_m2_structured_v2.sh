#!/usr/bin/env bash
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_structured_v2_job_ids.env
test "$(git rev-parse HEAD)" = "$M2_V2_COMMIT"
test -z "$(git status --porcelain)"
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=8G --time=00:30:00 \
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh \
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL \
  env PYTHONPATH=src .venv/bin/python scripts/run_m2_structured_v2.py verify \
  --phase readiness --job-id "${READINESS_JOB_ID:?Readiness was not submitted}"
echo 'PASS: all three v2 stages recursively verified. M2 scientific work is not yet complete.'
