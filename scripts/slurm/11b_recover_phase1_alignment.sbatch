#!/usr/bin/env bash
#SBATCH --job-name=tehr-phase1-recovery
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
SOURCE_ALIGNMENT_JOB_ID="${SOURCE_ALIGNMENT_JOB_ID:?Set SOURCE_ALIGNMENT_JOB_ID=407 before submission}"
if [[ "${SOURCE_ALIGNMENT_JOB_ID}" != "407" ]]; then
  echo "This evidence-bound recovery accepts only SOURCE_ALIGNMENT_JOB_ID=407." >&2
  exit 1
fi

SOURCE_GIT_COMMIT="f9a370c7fe1cb13edd126128cfddc79100852913"
SOURCE_STARTED_AT="2026-08-20T16:42:02Z"
SOURCE_ENDED_AT="2026-08-20T19:06:03Z"
EXPECTED_REPORT_SIZE=362211
EXPECTED_REPORT_MTIME="2026-08-20T19:05:58Z"
EXPECTED_NOTEBOOK_SIZE=448911
EXPECTED_NOTEBOOK_MTIME="2026-08-20T19:05:59Z"

CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
RECOVERY_JOB_ID="${SLURM_JOB_ID:?Recovery must run as a scheduled Slurm job}"
STATUS_DIR="${PROJECT_ROOT}/outputs/private/run_status/${RECOVERY_JOB_ID}"
SOURCE_STATUS_DIR="${PROJECT_ROOT}/outputs/private/run_status/${SOURCE_ALIGNMENT_JOB_ID}"
SOURCE_FAILED_SENTINEL="${SOURCE_STATUS_DIR}/FAILED.json"
VALIDATION_REPORT="${STATUS_DIR}/recovery_validation.json"
REPORT_PATH="${PROJECT_ROOT}/reports/generated/phase1_protocol_review/protocol_alignment.json"
NOTEBOOK_OUTPUT="${PROJECT_ROOT}/outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb"
SOURCE_ENVIRONMENT_AUDIT="${SOURCE_STATUS_DIR}/environment_audit.json"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
RUN_SUCCEEDED=0

if [[ "${RECOVERY_JOB_ID}" == "${SOURCE_ALIGNMENT_JOB_ID}" ]]; then
  echo "Recovery job ID must differ from source job ID 407." >&2
  exit 1
fi
for required_file in "${CONTAINER_IMAGE}" "${PYTHON}" "${SOURCE_FAILED_SENTINEL}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required recovery input: ${required_file##*/}" >&2
    exit 1
  fi
done

mkdir -p "${STATUS_DIR}"
cd "${PROJECT_ROOT}"
RECOVERY_GIT_COMMIT="$(git rev-parse HEAD)"
SOURCE_FAILED_SHA256="$(sha256sum "${SOURCE_FAILED_SENTINEL}" | awk '{print $1}')"
STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

run_in_container() {
  srun \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mounts=/data:/data \
    --container-workdir="${PROJECT_ROOT}" \
    --export=ALL \
    env PYTHONPATH=src "$@"
}

run_in_container "${PYTHON}" scripts/write_run_status.py RUNNING \
  --status-dir "${STATUS_DIR}" \
  --job-id "${RECOVERY_JOB_ID}" \
  --started-at "${STARTED_AT}" \
  --git-commit "${RECOVERY_GIT_COMMIT}" \
  --run-kind phase1_alignment_recovery \
  --environment-audit "outputs/private/run_status/${SOURCE_ALIGNMENT_JOB_ID}/environment_audit.json" \
  --notebook-output "outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb" \
  --source-job-id "${SOURCE_ALIGNMENT_JOB_ID}" \
  --source-git-commit "${SOURCE_GIT_COMMIT}" \
  --recovery-git-commit "${RECOVERY_GIT_COMMIT}" \
  --validation-report "outputs/private/run_status/${RECOVERY_JOB_ID}/recovery_validation.json" \
  --source-failed-sentinel-sha256 "${SOURCE_FAILED_SHA256}"

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
  run_in_container "${PYTHON}" scripts/write_run_status.py "${final_state}" \
    --status-dir "${STATUS_DIR}" \
    --ended-at "${ended_at}" \
    --exit-status "${exit_status}"
  exit "${exit_status}"
}
trap finalize_status EXIT

export PYTHONHASHSEED=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_in_container "${PYTHON}" scripts/validate_phase1_alignment_recovery.py \
  --repository-root "${PROJECT_ROOT}" \
  --report "${REPORT_PATH}" \
  --notebook "${NOTEBOOK_OUTPUT}" \
  --source-failed-sentinel "${SOURCE_FAILED_SENTINEL}" \
  --source-job-id "${SOURCE_ALIGNMENT_JOB_ID}" \
  --source-git-commit "${SOURCE_GIT_COMMIT}" \
  --recovery-git-commit "${RECOVERY_GIT_COMMIT}" \
  --source-started-at "${SOURCE_STARTED_AT}" \
  --source-ended-at "${SOURCE_ENDED_AT}" \
  --expected-report-size "${EXPECTED_REPORT_SIZE}" \
  --expected-report-mtime "${EXPECTED_REPORT_MTIME}" \
  --expected-notebook-size "${EXPECTED_NOTEBOOK_SIZE}" \
  --expected-notebook-mtime "${EXPECTED_NOTEBOOK_MTIME}" \
  --output "${VALIDATION_REPORT}"

run_in_container "${PYTHON}" scripts/review_phase1_results.py --report "${REPORT_PATH}"

if [[ ! -s "${VALIDATION_REPORT}" ]]; then
  echo "Recovery validation report is missing or empty." >&2
  exit 1
fi
if [[ "$(sha256sum "${SOURCE_FAILED_SENTINEL}" | awk '{print $1}')" != "${SOURCE_FAILED_SHA256}" ]]; then
  echo "Original job-407 FAILED.json changed during recovery." >&2
  exit 1
fi

RUN_SUCCEEDED=1
