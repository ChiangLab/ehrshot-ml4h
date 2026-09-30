#!/usr/bin/env bash
# Read-only status; safe on the login node and from any directory.
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$(id -un)" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
test -s outputs/private/m2_selection_v3_job_ids.env || {
  echo 'No selection-v3 submission record exists yet.'
  exit 1
}
source outputs/private/m2_selection_v3_job_ids.env
export M2_SELECTION_SHARDS="${M2_SELECTION_SHARDS:-0}"
export M2_SELECTION_CANARY_JOB="${M2_SELECTION_CANARY_JOB:-}"
export M2_SELECTION_CONFIRMATION_JOB="${M2_SELECTION_CONFIRMATION_JOB:-}"
export M2_SELECTION_READINESS_JOB="${M2_SELECTION_READINESS_JOB:-}"
export M2_SELECTION_ARRAY="${M2_SELECTION_ARRAY:-}"
export M2_SELECTION_SEAL_JOB="${M2_SELECTION_SEAL_JOB:-}"
export M2_SELECTION_EMBED_ARRAY="${M2_SELECTION_EMBED_ARRAY:-}"
export M2_SELECTION_COMMIT M2_SELECTION_SHARDS M2_SELECTION_CANARY_JOB
export M2_SELECTION_CONFIRMATION_JOB M2_SELECTION_READINESS_JOB
export M2_SELECTION_ARRAY M2_SELECTION_SEAL_JOB M2_SELECTION_EMBED_ARRAY
if [[ -s outputs/private/m2_selection_v3_fit_job_ids.env ]]; then
  source outputs/private/m2_selection_v3_fit_job_ids.env
  export M2_SELECTION_FIT_JOB
fi
python3 - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

root = Path("outputs/private/run_status")
commit = os.environ["M2_SELECTION_COMMIT"]

def state(job: str) -> str:
    if not job:
        return "NOT_SUBMITTED"
    complete = root / job / "COMPLETE.json"
    failed = root / job / "FAILED.json"
    if complete.exists() and failed.exists():
        return "CONFLICT"
    if failed.exists():
        return "FAILED"
    if complete.exists():
        value = json.loads(complete.read_text())
        if value.get("state") == "COMPLETE" and value.get("exit_status") == 0 and value.get("git_commit") == commit:
            return "COMPLETE"
        return "INVALID"
    return "NO_FINAL_SENTINEL"

for name in (
    "M2_SELECTION_CANARY_JOB",
    "M2_SELECTION_CONFIRMATION_JOB",
    "M2_SELECTION_READINESS_JOB",
):
    job = os.environ[name]
    print(name, job, state(job))
array = os.environ["M2_SELECTION_ARRAY"]
shards = int(os.environ["M2_SELECTION_SHARDS"])
print(
    "Generation:",
    dict(Counter(state(f"{array}_{index}") for index in range(shards)))
    if array and shards
    else "NOT_SUBMITTED",
)
print("M2_SELECTION_SEAL_JOB", os.environ["M2_SELECTION_SEAL_JOB"], state(os.environ["M2_SELECTION_SEAL_JOB"]))
embed = os.environ["M2_SELECTION_EMBED_ARRAY"]
if embed:
    for index in range(4):
        print(f"Embedding {embed}_{index}:", state(f"{embed}_{index}"))
else:
    print("Embedding: NOT_SUBMITTED")
fit = os.environ.get("M2_SELECTION_FIT_JOB")
print("M2_SELECTION_FIT_JOB", fit or "NOT_SUBMITTED", state(fit) if fit else "")
halt = Path("outputs/private/final/m2_selection_v3_reference_v1/HALT.json")
if halt.exists():
    print("STOP: v3 circuit breaker is set; preserve and inspect evidence.")
print("COMPLETE sentinels are operational evidence; final semantic verification is still required.")
PY
