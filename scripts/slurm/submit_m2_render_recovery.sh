#!/usr/bin/env bash
# Exactly one read-only replay and one dependent frozen-sample confirmation.
set -euo pipefail
umask 077
export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
cd "$TEMPORAL_EHR_PROJECT_ROOT"
test -z "$(git status --porcelain)" || { echo 'STOP: checkout is dirty'; exit 1; }
QUEUE="$(squeue -h -u "$(id -un)" -o '%i %T %j')"
test -z "$QUEUE" || { echo 'STOP: inspect active jobs first'; echo "$QUEUE"; exit 1; }
test -s outputs/private/run_status/982/FAILED.json
test -s outputs/private/recovery/m2_structured_v1/982/manifest.json
export M2_RENDER_COMMIT="$(git rev-parse HEAD)"
mkdir -p outputs/private/submissions
RECORD="outputs/private/submissions/m2-render-${M2_RENDER_COMMIT}.env"
if [[ -e "$RECORD" ]]; then
  echo 'STOP: submission already recorded; inspect before any retry'
  cat "$RECORD"
  exit 1
fi
(set -o noclobber; printf 'M2_RENDER_COMMIT=%q\n' "$M2_RENDER_COMMIT" > "$RECORD")
ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_render_job_ids.env
export M2_RENDER_PHASE=replay
REPLAY_JOB_ID="$(sbatch --parsable --time=00:30:00 --export=ALL scripts/slurm/24f_m2_render_recovery.sbatch)"
REPLAY_JOB_ID="${REPLAY_JOB_ID%%;*}"
[[ "$REPLAY_JOB_ID" =~ ^[0-9]+$ ]]
printf 'REPLAY_JOB_ID=%q\n' "$REPLAY_JOB_ID" >> "$RECORD"
export M2_RENDER_PHASE=confirmation
export M2_RENDER_PARENT="$REPLAY_JOB_ID"
CONFIRMATION_JOB_ID="$(sbatch --parsable --time=04:00:00 --kill-on-invalid-dep=yes \
  --dependency="afterok:${REPLAY_JOB_ID}" --export=ALL scripts/slurm/24f_m2_render_recovery.sbatch)"
CONFIRMATION_JOB_ID="${CONFIRMATION_JOB_ID%%;*}"
[[ "$CONFIRMATION_JOB_ID" =~ ^[0-9]+$ ]]
printf 'CONFIRMATION_JOB_ID=%q\n' "$CONFIRMATION_JOB_ID" >> "$RECORD"
cat "$RECORD"
echo 'Replay then independent confirmation submitted. No full M2 graph submitted.'
