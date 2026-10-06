#!/usr/bin/env bash
# Explicit gated launch and post-review finalize for M2 selection-v4.
set -euo pipefail
umask 077
MODE="${1:-start}"
[[ "$MODE" = start || "$MODE" = resume || "$MODE" = finalize ]] || {
  echo 'Use: start, resume, or finalize'
  exit 1
}
export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
cd "$TEMPORAL_EHR_PROJECT_ROOT"
test -z "$(git status --porcelain)" || { echo 'STOP: checkout is dirty'; exit 1; }
QUEUE="$(squeue -h -u "$(id -un)" -o '%i %T %j')"
test -z "$QUEUE" || { echo 'STOP: review active jobs first'; echo "$QUEUE"; exit 1; }
export M2_SELECTION_COMMIT="$(git rev-parse HEAD)"
PRIVATE=outputs/private/final/m2_selection_v4_reference_v1
PUBLIC=reports/generated/final/m2_selection_v4_reference_v1
if [[ "$MODE" = start && -e "$PRIVATE/PLAN.json" ]]; then
  echo 'STOP: v4 plan exists; inspect status before an explicit resume'
  exit 1
fi
if [[ "$MODE" = resume ]]; then
  test -s "$PRIVATE/PLAN.json"
  python3 - <<'PY'
import json
from pathlib import Path
for phase in ("canary", "confirmation"):
    for path in Path("reports/generated/engineering/m2_selection_v4").glob(f"{phase}-*.json"):
        if json.loads(path.read_text()).get("status") != "PASS":
            raise SystemExit("STOP: failed v4 gate requires evidence review, not a blind retry")
PY
fi
test ! -e "$PRIVATE/HALT.json" || {
  echo 'STOP: selection-v4 circuit breaker is set; preserve and review it'
  exit 1
}
RUN=(
  srun --partition=batch --nodelist=dgit00038673 --cpus-per-task=2 --mem=32G
  --time=01:00:00
  --container-image=/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh
  --container-mounts=/data:/data --container-workdir="$PWD" --export=ALL
  env PYTHONPATH=src .venv/bin/python scripts/run_m2_selection_v4.py
)
mkdir -p outputs/private/submissions
STAMP="$(date -u +%Y%m%dT%H%M%S)-$$"
RECORD="outputs/private/submissions/m2-selection-v4-${MODE}-${STAMP}.env"
(set -o noclobber; printf 'M2_SELECTION_COMMIT=%q\n' "$M2_SELECTION_COMMIT" > "$RECORD")
record() {
  [[ "$2" =~ ^[0-9]+$ ]]
  printf '%s=%q\n' "$1" "$2" >> "$RECORD"
}

if [[ "$MODE" = finalize ]]; then
  test -s outputs/private/m2_selection_v4_job_ids.env
  EMBED_ARRAY="$(bash -c 'source outputs/private/m2_selection_v4_job_ids.env; printf "%s" "${M2_SELECTION_EMBED_ARRAY:?}"')"
  [[ "$EMBED_ARRAY" =~ ^[0-9]+$ ]]
  test -s "$PRIVATE/review/decisions.json"
  export M2_SELECTION_PHASE=fit M2_SELECTION_PARENT_ARRAY="$EMBED_ARRAY"
  FIT="$(sbatch --parsable --time=04:00:00 --mem=32G --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  FIT="${FIT%%;*}"
  record M2_SELECTION_FIT_JOB "$FIT"
  record M2_SELECTION_EMBED_ARRAY "$EMBED_ARRAY"
  ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_selection_v4_fit_job_ids.env
else
  if [[ "$MODE" = start ]]; then
    test ! -e "$PRIVATE/PLAN.json" || {
      echo 'STOP: v4 plan exists; use resume only after reading the status output'
      exit 1
    }
  else
    test -s "$PRIVATE/PLAN.json"
  fi
  "${RUN[@]}" plan
  SHARDS="$(python3 -c 'import json; print(json.load(open("reports/generated/final/m2_selection_v4_reference_v1/plan.json"))["shards"])')"
  [[ "$SHARDS" =~ ^[0-9]+$ ]] && (( SHARDS > 0 && SHARDS <= 999 ))
  LIMIT="$(scontrol show config | awk '$1 == "MaxArraySize" {n=$3} END {print n}')"
  [[ "$LIMIT" =~ ^[0-9]+$ ]] && (( SHARDS <= LIMIT )) || {
    echo 'STOP: site MaxArraySize is smaller than the frozen worklist'
    exit 1
  }
  record M2_SELECTION_SHARDS "$SHARDS"
  # Publish a new record only after preflight succeeds. Preserve previous attempts.
  ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_selection_v4_job_ids.env

  export M2_SELECTION_PHASE=canary
  CANARY="$(sbatch --parsable --time=04:00:00 --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  CANARY="${CANARY%%;*}"
  record M2_SELECTION_CANARY_JOB "$CANARY"

  export M2_SELECTION_PHASE=confirmation
  CONFIRMATION="$(sbatch --parsable --time=04:00:00 --dependency="afterok:$CANARY" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  CONFIRMATION="${CONFIRMATION%%;*}"
  record M2_SELECTION_CONFIRMATION_JOB "$CONFIRMATION"

  export M2_SELECTION_PHASE=readiness
  export M2_SELECTION_CANARY_JOB="$CANARY" M2_SELECTION_CONFIRMATION_JOB="$CONFIRMATION"
  READINESS="$(sbatch --parsable --time=01:00:00 --mem=16G --dependency="afterok:$CONFIRMATION" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  READINESS="${READINESS%%;*}"
  record M2_SELECTION_READINESS_JOB "$READINESS"
  export M2_SELECTION_READINESS_JOB="$READINESS"

  export M2_SELECTION_PHASE=shard
  ARRAY="$(sbatch --parsable --time=06:00:00 --array="0-$((SHARDS - 1))%2" --dependency="afterok:$READINESS" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  ARRAY="${ARRAY%%;*}"
  record M2_SELECTION_ARRAY "$ARRAY"

  export M2_SELECTION_PHASE=seal M2_SELECTION_PARENT_ARRAY="$ARRAY"
  export M2_SELECTION_READINESS_JOB="$READINESS"
  SEAL="$(sbatch --parsable --time=02:00:00 --mem=32G --dependency="afterok:$ARRAY" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  SEAL="${SEAL%%;*}"
  record M2_SELECTION_SEAL_JOB "$SEAL"

  export M2_SELECTION_PHASE=embed
  EMBED="$(sbatch --parsable --time=12:00:00 --mem=48G --gres=mps:1 --array=0-3%1 --dependency="afterok:$SEAL" --kill-on-invalid-dep=yes --export=ALL scripts/slurm/24k_m2_selection_v4.sbatch)"
  EMBED="${EMBED%%;*}"
  record M2_SELECTION_EMBED_ARRAY "$EMBED"
fi
cat "$RECORD"
echo 'Submitted. Monitor with scripts/slurm/m2_selection_v4_status.sh.'
echo 'The validation fit remains blocked until the 40-history manual review passes.'
