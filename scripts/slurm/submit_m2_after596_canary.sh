#!/usr/bin/env bash
# Submit ONLY a bounded engineering check. Never automatically launch 768 shards.
set -euo pipefail
umask 077
export TEMPORAL_EHR_PROJECT_ROOT="/data/shared73/projects/wdelery/temporal-ehr-ml4h"
export TEMPORAL_EHR_MODEL_ROOT="/data/shared73/projects/wdelery/model_cache/qwen3-embedding"
cd "${TEMPORAL_EHR_PROJECT_ROOT}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "STOP: commit and deploy the scoped repair before submitting." >&2
  exit 1
fi
# A scheduler error is a hard failure, not evidence of an empty queue.
QUEUE="$(squeue -h -u "$(id -un)" -o '%i %T %j')"
if [[ -n "${QUEUE}" ]]; then
  echo "STOP: inspect existing jobs before the recovery check:" >&2
  echo "${QUEUE}" >&2
  exit 1
fi
export M2_CANARY_SOURCE_COMMIT="$(git rev-parse HEAD)"
# No scientific config, prompt, model, decoding, source grouping or retention change.
git diff --exit-code 0d17bac55a561880086aae560640bfcdb719f021 -- configs/experiments/final_m2_m3.yaml
test -s "${TEMPORAL_EHR_MODEL_ROOT}/Qwen3-Embedding-0.6B/MODEL_ARTIFACT_MANIFEST.json"
mkdir -p outputs/private/submissions
RECORD="outputs/private/submissions/m2-after596-canary-${M2_CANARY_SOURCE_COMMIT}.env"
if [[ -e "${RECORD}" ]]; then
  echo "STOP: this source already has a canary submission; inspect it before retrying."
  cat "${RECORD}"
  exit 1
fi
# Reserve the record before sbatch so two launches cannot race.
(set -o noclobber; printf 'M2_CANARY_SOURCE_COMMIT=%q\n' "${M2_CANARY_SOURCE_COMMIT}" > "${RECORD}")
M2_CANARY_JOB_ID="$(sbatch --parsable --export=ALL scripts/slurm/24d_m2_after596_canary.sbatch)"
M2_CANARY_JOB_ID="${M2_CANARY_JOB_ID%%;*}"
[[ "${M2_CANARY_JOB_ID}" =~ ^[0-9]+$ ]]
printf 'M2_CANARY_JOB_ID=%q\n' "${M2_CANARY_JOB_ID}" >> "${RECORD}"
ln -sfn "submissions/$(basename "${RECORD}")" outputs/private/m2_after596_canary.env
cat "${RECORD}"
echo "No M2 embedding, fit, M3, or test job was submitted."
