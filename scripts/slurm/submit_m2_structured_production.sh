#!/usr/bin/env bash
# Explicit start/resume/finalize. No old launchers, held-out work, or automatic human approval.
set -euo pipefail
umask 077
MODE="${1:-start}"
[[ "$MODE" = start || "$MODE" = resume || "$MODE" = finalize ]] || { echo 'Use start, resume, or finalize'; exit 1; }
export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
cd "$TEMPORAL_EHR_PROJECT_ROOT"
test -z "$(git status --porcelain)" || { echo 'STOP: checkout is dirty'; exit 1; }
QUEUE="$(squeue -h -u "$(id -un)" -o '%i %T %j')"
test -z "$QUEUE" || { echo 'STOP: review active jobs first'; echo "$QUEUE"; exit 1; }
export M2_PRODUCTION_COMMIT="$(git rev-parse HEAD)"
PRIVATE=outputs/private/final/m2_structured_v2_reference_v1
PUBLIC=reports/generated/final/m2_structured_v2_reference_v1
test ! -e "$PRIVATE/HALT.json" || { echo 'STOP: rejected-input circuit breaker requires review; do not delete it'; exit 1; }
RUN=(srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=32G --time=01:00:00
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL
  env PYTHONPATH=src .venv/bin/python scripts/run_m2_structured_production.py)
mkdir -p outputs/private/submissions
STAMP="$(date -u +%Y%m%dT%H%M%S)-$$"
RECORD="outputs/private/submissions/m2-production-${MODE}-${STAMP}.env"
(set -o noclobber; printf 'M2_PRODUCTION_COMMIT=%q\n' "$M2_PRODUCTION_COMMIT" > "$RECORD")
record() { [[ "$2" =~ ^[0-9]+$ ]]; printf '%s=%q\n' "$1" "$2" >> "$RECORD"; }
if [[ "$MODE" = finalize ]]; then
  test -s outputs/private/m2_production_job_ids.env
  # Only read IDs. Do not allow a historical record to change the current commit.
  EMBED_ARRAY="$(bash -c 'source outputs/private/m2_production_job_ids.env; printf "%s" "${M2_PRODUCTION_EMBED_ARRAY:?}"')"
  [[ "$EMBED_ARRAY" =~ ^[0-9]+$ ]]
  test -s "$PRIVATE/review/decisions.json"
  export M2_PRODUCTION_PHASE=fit M2_PRODUCTION_PARENT_ARRAY="$EMBED_ARRAY"
  FIT="$(sbatch --parsable --time=04:00:00 --mem=32G --export=ALL scripts/slurm/24h_m2_structured_production.sbatch)"
  FIT="${FIT%%;*}"
  record M2_PRODUCTION_FIT_JOB "$FIT"
  record M2_PRODUCTION_EMBED_ARRAY "$EMBED_ARRAY"
  ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_production_fit_job_ids.env
else
  if [[ "$MODE" = start ]]; then
    test ! -e "$PRIVATE/PLAN.json" || { echo 'STOP: plan already exists; use resume after checking status'; exit 1; }
  else
    test -s "$PRIVATE/PLAN.json"
  fi
  # Recursive engineering verification + deterministic source freeze. No new LLM generations.
  "${RUN[@]}" plan
  SHARDS="$(python3 -c 'import json; print(json.load(open("reports/generated/final/m2_structured_v2_reference_v1/plan.json"))["shards"])')"
  [[ "$SHARDS" =~ ^[0-9]+$ ]] && (( SHARDS > 0 && SHARDS <= 999 ))
  LIMIT="$(scontrol show config | awk '$1 == "MaxArraySize" {n=$3} END {print n}')"
  [[ "$LIMIT" =~ ^[0-9]+$ ]] && (( SHARDS <= LIMIT )) || { echo 'STOP: site array limit insufficient'; exit 1; }
  record M2_PRODUCTION_SHARDS "$SHARDS"
  ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_production_job_ids.env
  export M2_PRODUCTION_PHASE=shard
  ARRAY="$(sbatch --parsable --array="0-$((SHARDS-1))%2" --export=ALL scripts/slurm/24h_m2_structured_production.sbatch)"
  ARRAY="${ARRAY%%;*}"; record M2_PRODUCTION_ARRAY "$ARRAY"
  export M2_PRODUCTION_PHASE=seal M2_PRODUCTION_PARENT_ARRAY="$ARRAY"
  SEAL="$(sbatch --parsable --time=02:00:00 --mem=32G --dependency="afterok:$ARRAY" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24h_m2_structured_production.sbatch)"
  SEAL="${SEAL%%;*}"; record M2_PRODUCTION_SEAL_JOB "$SEAL"
  export M2_PRODUCTION_PHASE=embed
  EMBED="$(sbatch --parsable --time=12:00:00 --mem=48G --gres=mps:1 --array=0-3%1 --dependency="afterok:$SEAL" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24h_m2_structured_production.sbatch)"
  EMBED="${EMBED%%;*}"; record M2_PRODUCTION_EMBED_ARRAY "$EMBED"
fi
cat "$RECORD"
echo 'Submitted. Monitor with scripts/slurm/m2_production_status.sh. Manual review is NOT automated.'
