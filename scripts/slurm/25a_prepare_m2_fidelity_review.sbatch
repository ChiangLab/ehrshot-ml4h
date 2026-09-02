#!/usr/bin/env bash
#SBATCH --job-name=tehr-m2-review-packet
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CONFIG="${PROJECT_ROOT}/configs/experiments/final_m2_m3.yaml"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize final_m2_manual_review_prepare "${SLURM_JOB_ID:?Slurm job ID required}"
export FINAL_M23_GIT_COMMIT="${FINAL_M1_GIT_COMMIT}"
final_container_python "${PYTHON}" scripts/run_final_m23.py --config "${CONFIG}" hard-gates
final_container_python "${PYTHON}" scripts/review_m2_fidelity_manual.py prepare --config "${CONFIG}"
final_mark_success
