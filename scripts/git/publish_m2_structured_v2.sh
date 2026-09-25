#!/usr/bin/env bash
# Explicit user-invoked local publication; never stage papers, deliverables, or governed data.
set -euo pipefail
cd /Users/williamdelery/code/temporal-ehr-ml4h
test "$(git branch --show-current)" = main || { echo 'STOP: expected main'; exit 1; }
git diff --cached --quiet || { echo 'STOP: existing staged changes require review'; exit 1; }
FILES=(
  AGENTS.md docs/CURRENT_SCIENTIFIC_PROTOCOL.md docs/PROJECT_STATE.md docs/DECISION_LOG.md
  docs/NEXT_ACTION.md docs/DGX_BATCH_RUNBOOK.md docs/PROTOCOL_DEVIATIONS.md docs/RESULTS_REGISTRY.md
  docs/M2_STRUCTURED_V2_AFTER_986.md
  src/temporal_ehr/final_m23/structured_v2.py
  src/temporal_ehr/final_m23/structured_v2_recovery.py
  scripts/run_m2_structured_v2.py scripts/slurm/24g_m2_structured_v2.sbatch
  scripts/slurm/submit_m2_structured_v2.sh scripts/slurm/m2_structured_v2_status.sh
  scripts/slurm/verify_m2_structured_v2.sh scripts/git/publish_m2_structured_v2.sh
  tests/test_m2_structured_v2.py
)
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python scripts/check_project_context.py
for script in "${FILES[@]}"; do
  case "$script" in *.sh|*.sbatch) bash -n "$script" ;; esac
done
command -v rg >/dev/null
bash scripts/git/pre_push_safety.sh
git add -- "${FILES[@]}"
git diff --cached --check
git --no-pager diff --cached --stat
bash scripts/git/pre_push_safety.sh
git commit -m "Version M2 generation repair after confirmation 986"
git push origin main
git rev-parse HEAD
