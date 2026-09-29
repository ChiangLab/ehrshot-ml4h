#!/usr/bin/env bash
# Read-only, stdlib-only status; safe on the login node, from any working directory.
set -euo pipefail
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h
squeue -u "$(id -un)" -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
test -s outputs/private/m2_production_job_ids.env || { echo 'No production submission record yet'; exit 1; }
source outputs/private/m2_production_job_ids.env
export M2_PRODUCTION_ARRAY M2_PRODUCTION_SHARDS M2_PRODUCTION_SEAL_JOB M2_PRODUCTION_EMBED_ARRAY M2_PRODUCTION_COMMIT
if [[ -s outputs/private/m2_production_fit_job_ids.env ]]; then
  source outputs/private/m2_production_fit_job_ids.env
  export M2_PRODUCTION_FIT_JOB
fi
python3 - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

root = Path('outputs/private/run_status')
def state(job):
    complete, failed = root / job / 'COMPLETE.json', root / job / 'FAILED.json'
    if complete.exists() and failed.exists():
        return 'CONFLICT'
    if failed.exists():
        return 'FAILED'
    if complete.exists():
        value = json.loads(complete.read_text())
        return 'COMPLETE' if (value.get('state') == 'COMPLETE' and value.get('exit_status') == 0
            and value.get('git_commit') == os.environ['M2_PRODUCTION_COMMIT']) else 'INVALID'
    return 'NO_FINAL_SENTINEL'

array = os.environ.get('M2_PRODUCTION_ARRAY', '')
count = int(os.environ.get('M2_PRODUCTION_SHARDS') or 0)
print('Generation:', dict(Counter(state(f'{array}_{i}') for i in range(count))))
for name in ('M2_PRODUCTION_SEAL_JOB', 'M2_PRODUCTION_FIT_JOB'):
    job = os.environ.get(name)
    print(name, job or 'NOT_SUBMITTED', state(job) if job else '')
array = os.environ.get('M2_PRODUCTION_EMBED_ARRAY')
if array:
    for index in range(4):
        print(f'Embedding {array}_{index}:', state(f'{array}_{index}'))
halt = Path('outputs/private/final/m2_structured_v2_reference_v1/HALT.json')
if halt.exists():
    print('STOP: a summary failed fidelity. Further generation is blocked; preserve and review evidence.')
print('No final sentinel does not distinguish pending, active, canceled, or never started; consult queue/logs.')
print('COMPLETE job sentinels still require the final semantic verifier.')
PY
