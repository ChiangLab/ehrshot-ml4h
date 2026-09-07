#!/usr/bin/env bash
#SBATCH --job-name=tehr-p4-overnight-launch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=batch
#SBATCH --nodelist=dgit00038673
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --mail-user=wdelery@mednet.ucla.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="${TEMPORAL_EHR_PROJECT_ROOT:-/data/shared73/projects/wdelery/temporal-ehr-ml4h}"
CONTAINER_IMAGE="/local_storage/enroot/cache/nvidia+pytorch+26.04-py3.sqsh"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
CAPACITY_JOB_ID="${PROMPT4_CAPACITY_JOB_ID:?Capacity job ID is required}"

source "${PROJECT_ROOT}/scripts/slurm/final_job_common.sh"
final_initialize prompt4_m2_overnight_launcher "${SLURM_JOB_ID:?Slurm job ID required}"

cd "${PROJECT_ROOT}"
SOURCE_GIT_COMMIT="$(git rev-parse HEAD)"
CAPACITY_ENV="outputs/private/prompt4_m2_capacity_v7_job_id.env"
test -s "${CAPACITY_ENV}"
# shellcheck disable=SC1090
source "${CAPACITY_ENV}"
if [[ "${CAPACITY_JOB_ID}" != "${PROMPT4_CAPACITY_JOB_ID}" || \
      "${CAPACITY_SOURCE_GIT_COMMIT}" != "${SOURCE_GIT_COMMIT}" || \
      ! -s "outputs/private/run_status/${CAPACITY_JOB_ID}/COMPLETE.json" || \
      -e "outputs/private/run_status/${CAPACITY_JOB_ID}/FAILED.json" ]]; then
  echo "The overnight launcher requires same-source COMPLETE capacity evidence." >&2
  exit 1
fi

bash scripts/slurm/submit_prompt4_m2_fidelity_v7.sh
# shellcheck disable=SC1091
source outputs/private/prompt4_reference_job_ids.env

export PROMPT4_M3_EVIDENCE_SOURCE_GIT_COMMIT="${M3_EVIDENCE_SOURCE_GIT_COMMIT}"
export PROMPT4_M2_RECOVERY_REPORT="${M2_RECOVERY_REPORT}"
export PROMPT4_M2_RECOVERY_REPORT_SHA256="${M2_RECOVERY_REPORT_SHA256}"
M3_REFERENCE_REPORT_JOB_ID="$(
  sbatch --parsable \
    --mail-user="${SLURM_MAIL_USER}" \
    --mail-type=END,FAIL \
    --export=ALL \
    scripts/slurm/27a_m3_reference_report.sbatch
)"
M3_REFERENCE_REPORT_JOB_ID="${M3_REFERENCE_REPORT_JOB_ID%%;*}"

mkdir -p outputs/private/submissions
LAUNCH_FILE="outputs/private/submissions/prompt4-overnight-launched-${SLURM_JOB_ID}.env"
TEMPORARY_FILE="$(mktemp "${LAUNCH_FILE}.tmp.XXXXXX")"
{
  printf 'SOURCE_GIT_COMMIT=%q\n' "${SOURCE_GIT_COMMIT}"
  printf 'CAPACITY_JOB_ID=%q\n' "${CAPACITY_JOB_ID}"
  printf 'LAUNCHER_JOB_ID=%q\n' "${SLURM_JOB_ID}"
  printf 'M2_ARRAY_JOB_ID=%q\n' "${M2_ARRAY_JOB_ID}"
  printf 'M2_EMBED_ARRAY_JOB_ID=%q\n' "${M2_EMBED_ARRAY_JOB_ID}"
  printf 'M2_REVIEW_PREP_JOB_ID=%q\n' "${M2_REVIEW_PREP_JOB_ID}"
  printf 'M3_REFERENCE_REPORT_JOB_ID=%q\n' "${M3_REFERENCE_REPORT_JOB_ID}"
} > "${TEMPORARY_FILE}"
mv "${TEMPORARY_FILE}" "${LAUNCH_FILE}"
ln -sfn "submissions/$(basename "${LAUNCH_FILE}")" \
  outputs/private/prompt4_overnight_launched_job_ids.env

printf 'M2_ARRAY_JOB_ID=%s\n' "${M2_ARRAY_JOB_ID}"
printf 'M2_EMBED_ARRAY_JOB_ID=%s\n' "${M2_EMBED_ARRAY_JOB_ID}"
printf 'M2_REVIEW_PREP_JOB_ID=%s\n' "${M2_REVIEW_PREP_JOB_ID}"
printf 'M3_REFERENCE_REPORT_JOB_ID=%s\n' "${M3_REFERENCE_REPORT_JOB_ID}"
final_mark_success
