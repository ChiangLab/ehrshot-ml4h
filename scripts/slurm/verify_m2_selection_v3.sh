#!/usr/bin/env bash
# Read-only semantic verification after the manual gate and fit complete.
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_selection_v3_job_ids.env
source outputs/private/m2_selection_v3_fit_job_ids.env
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
  .venv/bin/python scripts/run_m2_selection_recovery.py verify \
    --array "$M2_SELECTION_EMBED_ARRAY" \
    --fit-job "$M2_SELECTION_FIT_JOB"
echo 'PASS: selection-v3 M2 reference result recursively verified; test remained closed.'
