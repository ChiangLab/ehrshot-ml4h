#!/usr/bin/env bash
#SBATCH --job-name=tehr-m1-test-freeze
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m1_submission.yaml"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_m1_submission_freeze "${SLURM_JOB_ID:?Slurm job ID required}"
final_container_python "${PYTHON}" scripts/run_final_submission.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/run_final_submission.py --config "${CONFIG}" freeze
final_container_python "${PYTHON}" scripts/review_final_submission.py --config "${CONFIG}" --require-frozen
final_mark_success
