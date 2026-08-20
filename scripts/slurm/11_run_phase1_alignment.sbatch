#!/usr/bin/env bash
#SBATCH --job-name=tehr-phase1-alignment
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH before sbatch --export=ALL}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT before sbatch --export=ALL}"
COHORT_PATH="${EHRSHOT_COHORT_PATH:-${PROJECT_ROOT}/outputs/private/cohorts/primary_panel.parquet}"
QWEN_PATH="${QWEN_EMBEDDING_PATH:?Set QWEN_EMBEDDING_PATH to the pinned local tokenizer directory}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
JOB_ID="${SLURM_JOB_ID:-manual}"
STATUS_DIR="${PROJECT_ROOT}/outputs/private/run_status/${JOB_ID}"
RUNTIME_ROOT="${PROJECT_ROOT}/outputs/private/runtime/${JOB_ID}"
ENVIRONMENT_AUDIT="${STATUS_DIR}/environment_audit.json"
NOTEBOOK_OUTPUT="${PROJECT_ROOT}/outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb"
REPORT_PATH="${PROJECT_ROOT}/reports/generated/phase1_protocol_review/protocol_alignment.json"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUN_SUCCEEDED=0

for required_file in \
  "${EVENT_PATH}" \
  "${COHORT_PATH}" \
  "${CONTAINER_IMAGE}" \
  "${PROJECT_ROOT}/notebooks/03_phase1_protocol_review.ipynb" \
  "${PYTHON}"
do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file##*/}" >&2
    exit 1
  fi
done
for required_directory in "${PROJECT_ROOT}" "${ASSET_ROOT}" "${QWEN_PATH}"
do
  if [[ ! -d "${required_directory}" ]]; then
    echo "Missing required directory: ${required_directory##*/}" >&2
    exit 1
  fi
done

mkdir -p \
  "${STATUS_DIR}" \
  "${RUNTIME_ROOT}/ipython" \
  "${RUNTIME_ROOT}/jupyter" \
  "${RUNTIME_ROOT}/xdg" \
  "${RUNTIME_ROOT}/matplotlib" \
  "${RUNTIME_ROOT}/huggingface" \
  "${PROJECT_ROOT}/outputs/private/executed_notebooks"

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

cd "${PROJECT_ROOT}"
PYTHONPATH=src "${PYTHON}" scripts/write_run_status.py RUNNING \
  --status-dir "${STATUS_DIR}" \
  --job-id "${JOB_ID}" \
  --started-at "${STARTED_AT}" \
  --git-commit "${GIT_COMMIT}" \
  --environment-audit "outputs/private/run_status/${JOB_ID}/environment_audit.json" \
  --notebook-output "outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb"

finalize_status() {
  exit_status=$?
  trap - EXIT
  final_state="FAILED"
  if [[ "${exit_status}" -eq 0 && "${RUN_SUCCEEDED}" -eq 1 ]]; then
    final_state="COMPLETE"
  elif [[ "${exit_status}" -eq 0 ]]; then
    exit_status=1
  fi
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  PYTHONPATH=src "${PYTHON}" scripts/write_run_status.py "${final_state}" \
    --status-dir "${STATUS_DIR}" \
    --ended-at "${ended_at}" \
    --exit-status "${exit_status}"
  exit "${exit_status}"
}
trap finalize_status EXIT

export PYTHONPATH=src
export PYTHONHASHSEED=0
export EHRSHOT_DATA_PATH="${EVENT_PATH}"
export EHRSHOT_ASSET_ROOT="${ASSET_ROOT}"
export EHRSHOT_COHORT_PATH="${COHORT_PATH}"
export QWEN_EMBEDDING_PATH="${QWEN_PATH}"
export IPYTHONDIR="${RUNTIME_ROOT}/ipython"
export JUPYTER_CONFIG_DIR="${RUNTIME_ROOT}/jupyter"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"
export MPLCONFIGDIR="${RUNTIME_ROOT}/matplotlib"
export HF_HOME="${RUNTIME_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${RUNTIME_ROOT}/huggingface/hub"
export TRANSFORMERS_CACHE="${RUNTIME_ROOT}/huggingface/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  "${PYTHON}" scripts/audit_environment.py --output "${ENVIRONMENT_AUDIT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts=/data:/data \
  --container-workdir="${PROJECT_ROOT}" \
  --export=ALL \
  "${PYTHON}" scripts/execute_notebook.py \
  notebooks/03_phase1_protocol_review.ipynb \
  "${NOTEBOOK_OUTPUT}"

if [[ ! -s "${REPORT_PATH}" || ! "${REPORT_PATH}" -nt "${STATUS_DIR}/RUNNING.json" ]]; then
  echo "Notebook 03 did not produce a new aggregate protocol-alignment report." >&2
  exit 1
fi
if [[ ! -s "${NOTEBOOK_OUTPUT}" || ! "${NOTEBOOK_OUTPUT}" -nt "${STATUS_DIR}/RUNNING.json" ]]; then
  echo "Notebook 03 did not produce a new executed notebook." >&2
  exit 1
fi

PYTHONPATH=src "${PYTHON}" scripts/validate_executed_notebook.py "${NOTEBOOK_OUTPUT}"
PYTHONPATH=src "${PYTHON}" scripts/review_phase1_results.py --report "${REPORT_PATH}"
RUN_SUCCEEDED=1
