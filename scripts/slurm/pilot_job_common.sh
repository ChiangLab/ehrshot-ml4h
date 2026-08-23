#!/usr/bin/env bash

set -euo pipefail

pilot_initialize() {
  PILOT_RUN_KIND="$1"
  PILOT_STATUS_ID="$2"
  PILOT_RUN_SUCCEEDED=0
  PILOT_STATUS_DIR="${PROJECT_ROOT}/outputs/private/run_status/${PILOT_STATUS_ID}"
  PILOT_RUNTIME_ROOT="${PROJECT_ROOT}/outputs/private/runtime/${PILOT_STATUS_ID}"
  mkdir -p \
    "${PILOT_STATUS_DIR}" \
    "${PILOT_RUNTIME_ROOT}/ipython" \
    "${PILOT_RUNTIME_ROOT}/jupyter" \
    "${PILOT_RUNTIME_ROOT}/xdg" \
    "${PILOT_RUNTIME_ROOT}/matplotlib" \
    "${PILOT_RUNTIME_ROOT}/huggingface"
  cd "${PROJECT_ROOT}"
  PILOT_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  PILOT_GIT_COMMIT="$(git rev-parse HEAD)"
  export PILOT_GIT_COMMIT
  export PYTHONHASHSEED=0
  export IPYTHONDIR="${PILOT_RUNTIME_ROOT}/ipython"
  export JUPYTER_CONFIG_DIR="${PILOT_RUNTIME_ROOT}/jupyter"
  export XDG_CACHE_HOME="${PILOT_RUNTIME_ROOT}/xdg"
  export MPLCONFIGDIR="${PILOT_RUNTIME_ROOT}/matplotlib"
  export HF_HOME="${PILOT_RUNTIME_ROOT}/huggingface"
  export HUGGINGFACE_HUB_CACHE="${PILOT_RUNTIME_ROOT}/huggingface/hub"
  export TRANSFORMERS_CACHE="${PILOT_RUNTIME_ROOT}/huggingface/transformers"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false

  local status_args=(
    "${PYTHON}" scripts/write_run_status.py RUNNING
    --status-dir "${PILOT_STATUS_DIR}"
    --job-id "${PILOT_STATUS_ID}"
    --started-at "${PILOT_STARTED_AT}"
    --git-commit "${PILOT_GIT_COMMIT}"
    --run-kind "${PILOT_RUN_KIND}"
    --environment-audit ""
  )
  if [[ -n "${PILOT_NOTEBOOK_OUTPUT:-}" ]]; then
    status_args+=(--notebook-output "${PILOT_NOTEBOOK_OUTPUT}")
  fi
  pilot_container_python "${status_args[@]}"
  trap pilot_finalize EXIT
}

pilot_container_python() {
  srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/data:/data \
    --container-workdir="${PROJECT_ROOT}" \
    --export=ALL \
    env PYTHONPATH=src "$@"
}

pilot_finalize() {
  local exit_status=$?
  trap - EXIT
  local final_state="FAILED"
  if [[ "${exit_status}" -eq 0 && "${PILOT_RUN_SUCCEEDED}" -eq 1 ]]; then
    final_state="COMPLETE"
  elif [[ "${exit_status}" -eq 0 ]]; then
    exit_status=1
  fi
  local ended_at
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  pilot_container_python "${PYTHON}" scripts/write_run_status.py "${final_state}" \
    --status-dir "${PILOT_STATUS_DIR}" \
    --ended-at "${ended_at}" \
    --exit-status "${exit_status}"
  exit "${exit_status}"
}

pilot_mark_success() {
  PILOT_RUN_SUCCEEDED=1
}
