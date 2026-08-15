#!/usr/bin/env bash
#SBATCH --job-name=tehr-phase1-notebooks
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before sbatch --export=ALL}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before sbatch --export=ALL}"
QWEN_PATH="${QWEN_EMBEDDING_PATH:-${TEMPORAL_EHR_MODEL_ROOT:-}/Qwen3-Embedding-8B}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"

if [[ ! -f "${PROJECT_ROOT}/outputs/private/cohorts/primary_panel.parquet" ]]; then
  echo "Missing governed cohort. Rebuild it with:" >&2
  echo ".venv/bin/python -m temporal_ehr.cli build-cohort --asset-root \"\${EHRSHOT_ASSET_ROOT}\" --tasks lab_anemia new_acutemi guo_los --selection-seed 20260814 --private-output outputs/private/cohorts/primary_panel.parquet --allow-private-output" >&2
  exit 1
fi
if [[ -z "${QWEN_PATH}" || ! -d "${QWEN_PATH}" ]]; then
  echo "Set QWEN_EMBEDDING_PATH to the already-downloaded pinned Qwen tokenizer directory." >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs/slurm" "${PROJECT_ROOT}/outputs/private/executed_notebooks"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  env \
    PYTHONPATH=src \
    PYTHONHASHSEED=0 \
    EHRSHOT_DATA_PATH="${EVENT_PATH}" \
    EHRSHOT_ASSET_ROOT="${ASSET_ROOT}" \
    QWEN_EMBEDDING_PATH="${QWEN_PATH}" \
    bash -euo pipefail -c '
      for notebook in \
        00_ehrshot_foundation \
        01_count_baseline \
        02_serialization_qc
      do
        .venv/bin/python scripts/execute_notebook.py \
          "notebooks/${notebook}.ipynb" \
          "outputs/private/executed_notebooks/${notebook}.ipynb"
      done
    '
