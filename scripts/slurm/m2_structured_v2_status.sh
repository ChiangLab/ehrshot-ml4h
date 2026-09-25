#!/usr/bin/env bash
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
source outputs/private/m2_structured_v2_job_ids.env
printf 'Source: %s\n' "$M2_V2_COMMIT"
squeue -u "$(id -un)" -o '%.18i %.12P %.30j %.2t %.10M %.10l %R'
for phase in regression confirmation readiness; do
  case "$phase" in
    regression) job="${REGRESSION_JOB_ID:-}" ;;
    confirmation) job="${CONFIRMATION_JOB_ID:-}" ;;
    readiness) job="${READINESS_JOB_ID:-}" ;;
  esac
  if [[ ! "$job" =~ ^[0-9]+$ ]]; then echo "$phase: NOT SUBMITTED"; continue; fi
  directory="outputs/private/run_status/$job"
  if [[ -e "$directory/FAILED.json" ]]; then
    echo "$phase $job: FAILED or conflicting sentinels"
  elif [[ -s "$directory/COMPLETE.json" ]]; then
    echo "$phase $job: COMPLETE (semantic verification required)"
  else
    echo "$phase $job: no final sentinel; inspect queue (may be blocked/canceled)"
  fi
  report="reports/generated/engineering/m2_structured_v2/${phase}-${job}.json"
  [[ ! -s "$report" ]] || cat "$report"
  log="slurm-tehr-m2-v2-${job}.out"
  [[ ! -s "$log" ]] || tail -n 15 "$log"
done
