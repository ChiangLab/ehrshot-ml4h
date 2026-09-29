#!/usr/bin/env bash
# Explicit user-invoked publication; unrelated manuscript and deliverable changes stay untouched.
set -euo pipefail
cd /Users/williamdelery/code/temporal-ehr-ml4h
test "$(git branch --show-current)" = main
git diff --cached --quiet || { echo 'STOP: existing staged changes require review'; exit 1; }
FILES=(
  AGENTS.md docs/CURRENT_SCIENTIFIC_PROTOCOL.md docs/PROJECT_STATE.md docs/DECISION_LOG.md
  docs/NEXT_ACTION.md docs/DGX_BATCH_RUNBOOK.md docs/PROTOCOL_DEVIATIONS.md docs/RESULTS_REGISTRY.md
  docs/M2_STRUCTURED_V2_PRODUCTION.md
  docs/PROJECT_EXPLORER.md
  src/temporal_ehr/final_m23/structured_production.py src/temporal_ehr/final_m23/structured_reference.py
  src/temporal_ehr/project_explorer.py src/temporal_ehr/project_explorer_web.py
  scripts/run_m2_structured_production.py scripts/review_m2_structured_packet.py
  scripts/build_project_explorer.py scripts/serve_project_explorer.py
  scripts/slurm/24h_m2_structured_production.sbatch scripts/slurm/submit_m2_structured_production.sh
  scripts/slurm/m2_production_status.sh scripts/slurm/verify_m2_production.sh
  scripts/slurm/30_project_explorer.sbatch scripts/slurm/submit_project_explorer.sh
  scripts/slurm/project_explorer_status.sh
  scripts/git/publish_m2_structured_production.sh tests/test_m2_structured_production.py
  tests/test_project_explorer.py
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
git commit -m "Integrate gated structured M2 reference production"
git push origin main
git rev-parse HEAD
