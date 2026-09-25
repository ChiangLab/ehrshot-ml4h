#!/usr/bin/env bash
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_render_job_ids.env
printf 'Recovery source: %s\n' "${M2_RENDER_COMMIT:?}"
squeue -u "$(id -un)" -o '%.18i %.12P %.30j %.2t %.10M %.10l %R'
for phase in replay confirmation; do
  if [[ "$phase" == replay ]]; then job="${REPLAY_JOB_ID:-}"; else job="${CONFIRMATION_JOB_ID:-}"; fi
  if [[ ! "$job" =~ ^[0-9]+$ ]]; then echo "$phase: NOT SUBMITTED"; continue; fi
  status="outputs/private/run_status/$job"
  if [[ -s "$status/COMPLETE.json" && ! -e "$status/FAILED.json" ]]; then
    echo "$phase $job: COMPLETE (also require semantic verification)"
  elif [[ -e "$status/FAILED.json" ]]; then
    echo "$phase $job: FAILED or conflicting sentinels"
  else
    echo "$phase $job: NO FINAL SENTINEL; inspect queue/dependency"
  fi
  report="reports/generated/engineering/m2_render_v2/${phase}-${job}.json"
  if [[ -s "$report" ]]; then cat "$report"; fi
  log="slurm-tehr-m2-render-${job}.out"
  if [[ -f "$log" ]]; then tail -n 12 "$log"; fi
done
