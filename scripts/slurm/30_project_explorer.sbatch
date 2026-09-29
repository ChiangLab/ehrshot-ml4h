#!/usr/bin/env bash
#SBATCH --job-name=tehr-project-explorer
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-tehr-project-explorer-%j.out

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:?Set TEMPORAL_EHR_PROJECT_ROOT before submission}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before submission}"
SOURCE_COMMIT="${PROJECT_EXPLORER_SOURCE_COMMIT:?Set PROJECT_EXPLORER_SOURCE_COMMIT before submission}"
PORT="${PROJECT_EXPLORER_PORT:-8765}"
DATABASE="${PROJECT_EXPLORER_DATABASE:-outputs/private/inspection/temporal_ehr_project_explorer.duckdb}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh}"

cd "$PROJECT_ROOT"
test "$(git rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git status --porcelain --untracked-files=no)"
test -s "$EVENT_PATH"

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/data:/data \
  --container-workdir="$PROJECT_ROOT" \
  env PYTHONPATH=src \
  .venv/bin/python scripts/build_project_explorer.py \
  --repository-root "$PROJECT_ROOT" \
  --event-path "$EVENT_PATH" \
  --database "$DATABASE" \
  --source-commit "$SOURCE_COMMIT" \
  --deep-validate-embeddings

exec srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/data:/data \
  --container-workdir="$PROJECT_ROOT" \
  env PYTHONPATH=src \
  .venv/bin/python scripts/serve_project_explorer.py \
  --database "$DATABASE" \
  --host 127.0.0.1 \
  --port "$PORT"
