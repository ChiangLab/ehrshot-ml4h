#!/usr/bin/env bash
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_structured_job_ids.env
printf 'Source: %s\n' "${M2_STRUCTURED_COMMIT:?}"
squeue -u "$(id -un)" -o '%.18i %.12P %.30j %.2t %.10M %.10l %R'
for phase in development confirmation; do
  if [[ "$phase" == development ]]; then job="${DEVELOPMENT_JOB_ID:-}"; else job="${CONFIRMATION_JOB_ID:-}"; fi
  if [[ ! "$job" =~ ^[0-9]+$ ]]; then echo "$phase: NOT SUBMITTED"; continue; fi
  status="outputs/private/run_status/${job}"
  if [[ -s "$status/COMPLETE.json" && ! -e "$status/FAILED.json" ]]; then
    echo "$phase $job: COMPLETE; semantic verification still required"
  elif [[ -e "$status/FAILED.json" ]]; then
    echo "$phase $job: FAILED (or conflicting sentinels); do not launch production"
  else
    echo "$phase $job: NO FINAL SENTINEL; may be pending, active, or canceled before start"
  fi
  report="reports/generated/engineering/m2_structured_v1/${phase}-${job}.json"
  if [[ -s "$report" ]]; then cat "$report"; fi
  log="slurm-tehr-m2-structured-${job}.out"
  if [[ -f "$log" ]]; then tail -n 12 "$log"; fi
done
