#!/usr/bin/env bash
# Engineering only. No scientific array, embeddings, classifiers, M3, or test.
set -euo pipefail
umask 077
export TEMPORAL_EHR_PROJECT_ROOT=/data/shared73/projects/wdelery/temporal-ehr-ml4h
export TEMPORAL_EHR_MODEL_ROOT=/data/shared73/projects/wdelery/model_cache/qwen3-embedding
cd "${TEMPORAL_EHR_PROJECT_ROOT}"
test -z "$(git status --porcelain)" || { echo 'STOP: source checkout is dirty.'; exit 1; }
QUEUE="$(squeue -h -u "$(id -un)" -o '%i %T %j')"
test -z "${QUEUE}" || { echo 'STOP: inspect existing jobs before changing execution.'; echo "$QUEUE"; exit 1; }
test -s outputs/private/run_status/980/FAILED.json
test -s "${TEMPORAL_EHR_MODEL_ROOT}/Qwen3-Embedding-0.6B/MODEL_ARTIFACT_MANIFEST.json"
export M2_STRUCTURED_COMMIT="$(git rev-parse HEAD)"
mkdir -p outputs/private/submissions
RECORD="outputs/private/submissions/m2-structured-${M2_STRUCTURED_COMMIT}.env"
if [[ -e "${RECORD}" ]]; then
  echo 'STOP: this source already has a submission record. Inspect it; do not resubmit blindly.'
  cat "${RECORD}"
  exit 1
fi
(set -o noclobber; printf 'M2_STRUCTURED_COMMIT=%q\n' "$M2_STRUCTURED_COMMIT" > "$RECORD")
ln -sfn "submissions/$(basename "$RECORD")" outputs/private/m2_structured_job_ids.env
export M2_STRUCTURED_PHASE=development
DEVELOPMENT_JOB_ID="$(sbatch --parsable --export=ALL scripts/slurm/24e_m2_structured_gate.sbatch)"
DEVELOPMENT_JOB_ID="${DEVELOPMENT_JOB_ID%%;*}"
[[ "$DEVELOPMENT_JOB_ID" =~ ^[0-9]+$ ]]
printf 'DEVELOPMENT_JOB_ID=%q\n' "$DEVELOPMENT_JOB_ID" >> "$RECORD"
export M2_STRUCTURED_PHASE=confirmation
export M2_STRUCTURED_PARENT="$DEVELOPMENT_JOB_ID"
CONFIRMATION_JOB_ID="$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:${DEVELOPMENT_JOB_ID}" --export=ALL scripts/slurm/24e_m2_structured_gate.sbatch)"
CONFIRMATION_JOB_ID="${CONFIRMATION_JOB_ID%%;*}"
[[ "$CONFIRMATION_JOB_ID" =~ ^[0-9]+$ ]]
printf 'CONFIRMATION_JOB_ID=%q\n' "$CONFIRMATION_JOB_ID" >> "$RECORD"
cat "$RECORD"
echo 'Two bounded engineering jobs only. Full M2 remains gated on review of these reports.'
