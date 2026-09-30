#!/usr/bin/env bash
# Explicit user-invoked publication; unrelated manuscript/deliverable changes stay untouched.
set -euo pipefail
cd /Users/williamdelery/code/temporal-ehr-ml4h
test "$(git branch --show-current)" = main
git diff --cached --quiet || { echo 'STOP: existing staged changes require review'; exit 1; }
FILES=(
  AGENTS.md
  configs/recovery/m2_selection_v3_after_993.yaml
  docs/CURRENT_SCIENTIFIC_PROTOCOL.md
  docs/PROJECT_STATE.md
  docs/DECISION_LOG.md
  docs/NEXT_ACTION.md
  docs/DGX_BATCH_RUNBOOK.md
  docs/PROTOCOL_DEVIATIONS.md
  docs/RESULTS_REGISTRY.md
  docs/M2_SELECTION_V3_RECOVERY.md
  src/temporal_ehr/final_m23/structured_selection_v3.py
  src/temporal_ehr/final_m23/structured_selection_production.py
  src/temporal_ehr/final_m23/structured_selection_reference.py
  scripts/run_m2_selection_recovery.py
  scripts/review_m2_selection_v3_packet.py
  scripts/slurm/24j_m2_selection_recovery.sbatch
  scripts/slurm/submit_m2_selection_recovery.sh
  scripts/slurm/m2_selection_v3_status.sh
  scripts/slurm/verify_m2_selection_v3.sh
  scripts/git/publish_m2_selection_v3.sh
  tests/test_m2_selection_v3.py
)
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
.venv/bin/python scripts/check_project_context.py
for file in "${FILES[@]}"; do
  case "$file" in *.sh|*.sbatch) bash -n "$file" ;; esac
done
bash scripts/git/pre_push_safety.sh
git add -- "${FILES[@]}"
git diff --cached --check
git --no-pager diff --cached --stat
bash scripts/git/pre_push_safety.sh
git commit -m "Recover M2 with exact-source selection"
git push origin main
git rev-parse HEAD
