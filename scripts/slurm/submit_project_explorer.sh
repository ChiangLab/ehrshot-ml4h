#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-$PWD}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
PORT="${PROJECT_EXPLORER_PORT:-8765}"
DATABASE="${PROJECT_EXPLORER_DATABASE:-outputs/private/inspection/temporal_ehr_project_explorer.duckdb}"

cd "$PROJECT_ROOT"
SOURCE_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing submission: tracked DGX checkout is not clean." >&2
  exit 1
fi
if [[ ! -s "$EVENT_PATH" ]]; then
  echo "Refusing submission: EHRSHOT_DATA_PATH is missing or empty." >&2
  exit 1
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then
  echo "Refusing submission: PROJECT_EXPLORER_PORT must be 1024-65535." >&2
  exit 1
fi

JOB_ID="$(
  sbatch --parsable \
    --export="ALL,TEMPORAL_EHR_PROJECT_ROOT=$PROJECT_ROOT,EHRSHOT_DATA_PATH=$EVENT_PATH,PROJECT_EXPLORER_SOURCE_COMMIT=$SOURCE_COMMIT,PROJECT_EXPLORER_PORT=$PORT,PROJECT_EXPLORER_DATABASE=$DATABASE" \
    scripts/slurm/30_project_explorer.sbatch
)"
JOB_ID="${JOB_ID%%;*}"
mkdir -p outputs/private/submissions
RECORD="outputs/private/submissions/project-explorer-${JOB_ID}.env"
{
  printf 'PROJECT_EXPLORER_JOB_ID=%q\n' "$JOB_ID"
  printf 'PROJECT_EXPLORER_SOURCE_COMMIT=%q\n' "$SOURCE_COMMIT"
  printf 'PROJECT_EXPLORER_PORT=%q\n' "$PORT"
  printf 'PROJECT_EXPLORER_DATABASE=%q\n' "$DATABASE"
} > "$RECORD"
ln -sfn "submissions/$(basename "$RECORD")" outputs/private/project_explorer_job.env

echo "PROJECT_EXPLORER_JOB_ID=$JOB_ID"
echo "PROJECT_EXPLORER_PORT=$PORT"
echo "When the log reports project_explorer_ready, open a second Mac terminal and run:"
echo "ssh -N -L ${PORT}:127.0.0.1:${PORT} wdelery@10.45.62.90"
echo "Then open http://127.0.0.1:${PORT} in your Mac browser."
