#!/usr/bin/env bash

set -euo pipefail

cd "${TEMPORAL_EHR_PROJECT_ROOT:-$PWD}"
source outputs/private/project_explorer_job.env

squeue -j "$PROJECT_EXPLORER_JOB_ID" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R" || true

LOG="slurm-tehr-project-explorer-${PROJECT_EXPLORER_JOB_ID}.out"
if [[ -e "$LOG" ]]; then
  echo "===== RECENT LOG ====="
  tail -n 80 "$LOG"
fi

if squeue -h -j "$PROJECT_EXPLORER_JOB_ID" | grep -q . \
  && grep -q 'project_explorer_ready' "$LOG" 2>/dev/null; then
  echo "PASS: explorer is ready on DGX localhost port ${PROJECT_EXPLORER_PORT}."
  echo "Mac tunnel: ssh -N -L ${PROJECT_EXPLORER_PORT}:127.0.0.1:${PROJECT_EXPLORER_PORT} wdelery@10.45.62.90"
elif grep -q 'project_explorer_ready' "$LOG" 2>/dev/null; then
  echo "STOPPED: the explorer started successfully but is no longer running; submit a new 12-hour job."
elif ! squeue -h -j "$PROJECT_EXPLORER_JOB_ID" | grep -q .; then
  echo "FAILED OR ENDED BEFORE READY: inspect the log above before resubmitting."
else
  echo "WAIT: database validation/build or job startup is still in progress."
fi
