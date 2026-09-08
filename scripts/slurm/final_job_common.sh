#!/usr/bin/env bash

set -euo pipefail

final_initialize() {
  FINAL_RUN_KIND="$1"
  FINAL_STATUS_ID="$2"
  FINAL_RUN_SUCCEEDED=0
  FINAL_STATUS_DIR="${PROJECT_ROOT}/outputs/private/run_status/${FINAL_STATUS_ID}"
  FINAL_RUNTIME_ROOT="${PROJECT_ROOT}/outputs/private/runtime/${FINAL_STATUS_ID}"
  mkdir -p \
    "${FINAL_STATUS_DIR}" \
    "${FINAL_RUNTIME_ROOT}/ipython" \
    "${FINAL_RUNTIME_ROOT}/jupyter" \
    "${FINAL_RUNTIME_ROOT}/xdg" \
    "${FINAL_RUNTIME_ROOT}/matplotlib" \
    "${FINAL_RUNTIME_ROOT}/huggingface"
  cd "${PROJECT_ROOT}"
  FINAL_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  FINAL_M1_EXECUTION_GIT_COMMIT="$(git rev-parse HEAD)"
  FINAL_M1_GIT_COMMIT="${FINAL_M1_FROZEN_SOURCE_GIT_COMMIT:-${FINAL_M1_EXECUTION_GIT_COMMIT}}"
  if [[ ! "${FINAL_M1_EXECUTION_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ || \
        ! "${FINAL_M1_GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Invalid execution or frozen-source Git commit." >&2
    return 1
  fi
  if [[ -n "${FINAL_M1_FROZEN_SOURCE_GIT_COMMIT:-}" ]]; then
    FINAL_STATUS_SOURCE_GIT_COMMIT="${FINAL_STATUS_SOURCE_GIT_COMMIT:-${FINAL_M1_GIT_COMMIT}}"
    FINAL_STATUS_RECOVERY_GIT_COMMIT="${FINAL_STATUS_RECOVERY_GIT_COMMIT:-${FINAL_M1_EXECUTION_GIT_COMMIT}}"
  fi
  export FINAL_M1_EXECUTION_GIT_COMMIT
  export FINAL_M1_GIT_COMMIT
  export FINAL_STATUS_SOURCE_GIT_COMMIT
  export FINAL_STATUS_RECOVERY_GIT_COMMIT
  export PYTHONHASHSEED=0
  export IPYTHONDIR="${FINAL_RUNTIME_ROOT}/ipython"
  export JUPYTER_CONFIG_DIR="${FINAL_RUNTIME_ROOT}/jupyter"
  export XDG_CACHE_HOME="${FINAL_RUNTIME_ROOT}/xdg"
  export MPLCONFIGDIR="${FINAL_RUNTIME_ROOT}/matplotlib"
  export HF_HOME="${FINAL_RUNTIME_ROOT}/huggingface"
  export HUGGINGFACE_HUB_CACHE="${FINAL_RUNTIME_ROOT}/huggingface/hub"
  export TRANSFORMERS_CACHE="${FINAL_RUNTIME_ROOT}/huggingface/transformers"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false

  local status_args=(
    "${PYTHON}" scripts/write_run_status.py RUNNING
    --status-dir "${FINAL_STATUS_DIR}"
    --job-id "${FINAL_STATUS_ID}"
    --started-at "${FINAL_STARTED_AT}"
    --git-commit "${FINAL_M1_EXECUTION_GIT_COMMIT}"
    --run-kind "${FINAL_RUN_KIND}"
    --environment-audit ""
  )
  if [[ -n "${FINAL_NOTEBOOK_OUTPUT:-}" ]]; then
    status_args+=(--notebook-output "${FINAL_NOTEBOOK_OUTPUT}")
  fi
  if [[ -n "${FINAL_STATUS_SOURCE_JOB_ID:-}" ]]; then
    status_args+=(--source-job-id "${FINAL_STATUS_SOURCE_JOB_ID}")
  fi
  if [[ -n "${FINAL_STATUS_SOURCE_GIT_COMMIT:-}" ]]; then
    status_args+=(--source-git-commit "${FINAL_STATUS_SOURCE_GIT_COMMIT}")
  fi
  if [[ -n "${FINAL_STATUS_RECOVERY_GIT_COMMIT:-}" ]]; then
    status_args+=(--recovery-git-commit "${FINAL_STATUS_RECOVERY_GIT_COMMIT}")
  fi
  if [[ -n "${FINAL_STATUS_SOURCE_FAILED_SENTINEL_SHA256:-}" ]]; then
    status_args+=(
      --source-failed-sentinel-sha256
      "${FINAL_STATUS_SOURCE_FAILED_SENTINEL_SHA256}"
    )
  fi
  final_container_python "${status_args[@]}"
  trap final_finalize EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
  trap 'exit 129' HUP
}

final_container_python() {
  srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/data:/data \
    --container-workdir="${PROJECT_ROOT}" \
    --export=ALL \
    env PYTHONPATH=src "$@"
}

final_finalize() {
  local exit_status=$?
  trap - EXIT TERM INT HUP
  local final_state="FAILED"
  if [[ "${exit_status}" -eq 0 && "${FINAL_RUN_SUCCEEDED}" -eq 1 ]]; then
    final_state="COMPLETE"
  elif [[ "${exit_status}" -eq 0 ]]; then
    exit_status=1
  fi
  local ended_at
  ended_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! final_container_python "${PYTHON}" scripts/write_run_status.py "${final_state}" \
    --status-dir "${FINAL_STATUS_DIR}" \
    --ended-at "${ended_at}" \
    --exit-status "${exit_status}"; then
    PYTHONPATH=src "${PYTHON}" scripts/write_run_status.py "${final_state}" \
      --status-dir "${FINAL_STATUS_DIR}" \
      --ended-at "${ended_at}" \
      --exit-status "${exit_status}"
  fi
  exit "${exit_status}"
}

final_mark_success() {
  FINAL_RUN_SUCCEEDED=1
}
