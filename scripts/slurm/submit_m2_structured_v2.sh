#!/usr/bin/env bash
# Queue bounded engineering stages only. Preserve failures; never launch legacy M2.
set -euo pipefail
umask 077
export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
cd "$TEMPORAL_EHR_PROJECT_ROOT"
test -z "$(git status --porcelain)" || { echo 'STOP: checkout is dirty'; exit 1; }
QUEUE="$(squeue -h -u "$(id -un)" -o '%i %T %j')"
test -z "$QUEUE" || { echo 'STOP: active jobs must be reviewed'; echo "$QUEUE"; exit 1; }
test -s outputs/private/run_status/985/COMPLETE.json
test ! -e outputs/private/run_status/985/FAILED.json
test -s outputs/private/run_status/986/FAILED.json
test ! -e outputs/private/run_status/986/COMPLETE.json
export M2_V2_COMMIT="$(git rev-parse HEAD)"
mkdir -p outputs/private/submissions
RECORD="outputs/private/submissions/m2-structured-v2-${M2_V2_COMMIT}.env"
if [[ -e "$RECORD" ]]; then
  echo 'STOP: this submission is already recorded; inspect it instead of retrying'
  cat "$RECORD"
  exit 1
fi
(set -o noclobber; printf 'M2_V2_COMMIT=%q\n' "$M2_V2_COMMIT" > "$RECORD")
ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_structured_v2_job_ids.env
export M2_V2_PHASE=regression
REGRESSION_JOB_ID="$(sbatch --parsable --time=08:00:00 --export=ALL scripts/slurm/24g_m2_structured_v2.sbatch)"
REGRESSION_JOB_ID="${REGRESSION_JOB_ID%%;*}"
[[ "$REGRESSION_JOB_ID" =~ ^[0-9]+$ ]]
printf 'REGRESSION_JOB_ID=%q\n' "$REGRESSION_JOB_ID" >> "$RECORD"
export M2_V2_PHASE=confirmation M2_V2_PARENT="$REGRESSION_JOB_ID"
CONFIRMATION_JOB_ID="$(sbatch --parsable --time=04:00:00 --kill-on-invalid-dep=yes \
  --dependency="afterok:${REGRESSION_JOB_ID}" --export=ALL scripts/slurm/24g_m2_structured_v2.sbatch)"
CONFIRMATION_JOB_ID="${CONFIRMATION_JOB_ID%%;*}"
[[ "$CONFIRMATION_JOB_ID" =~ ^[0-9]+$ ]]
printf 'CONFIRMATION_JOB_ID=%q\n' "$CONFIRMATION_JOB_ID" >> "$RECORD"
export M2_V2_PHASE=readiness M2_V2_PARENT="$CONFIRMATION_JOB_ID"
READINESS_JOB_ID="$(sbatch --parsable --time=00:30:00 --kill-on-invalid-dep=yes \
  --dependency="afterok:${CONFIRMATION_JOB_ID}" --export=ALL scripts/slurm/24g_m2_structured_v2.sbatch)"
READINESS_JOB_ID="${READINESS_JOB_ID%%;*}"
[[ "$READINESS_JOB_ID" =~ ^[0-9]+$ ]]
printf 'READINESS_JOB_ID=%q\n' "$READINESS_JOB_ID" >> "$RECORD"
cat "$RECORD"
echo 'Queued: 64-input regression -> 32-input fresh confirmation -> verified readiness report.'
echo 'No full-cohort generation, embedding, fitting, M3, or held-out jobs submitted.'
