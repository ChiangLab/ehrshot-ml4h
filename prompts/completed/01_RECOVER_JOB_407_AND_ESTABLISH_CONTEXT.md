Prompt 1 — Recover job 407 and establish durable project context
Work in the current temporal-ehr-ml4h repository.

This task repairs the Phase 2 Slurm validator, validates the already-computed job-407 artifacts without rerunning Notebook 03, and creates a durable repository-local context system for subsequent Codex sessions.

Read completely before editing:

- README.md
- pyproject.toml
- docs/PROTOCOL_DEVIATIONS.md
- docs/SERIALIZATION_PROTOCOL.md
- docs/STATUS_REVIEW_2026-08-19.md
- prompts/02_AUDIT_PHASE1_AND_ALIGN_PROTOCOL.md
- scripts/slurm/11_run_phase1_alignment.sbatch
- scripts/validate_executed_notebook.py
- scripts/review_phase1_results.py
- scripts/write_run_status.py
- src/temporal_ehr/run_status.py
- /Users/williamdelery/Library/Mobile Documents/com~apple~CloudDocs/raptor_biodesign_fellowship/temporal representation/FINAL_TEMPORAL_EHR_PROJECT_PROTOCOL_2026-08-14.md

Evidence from DGX job 407:

- source Git commit: f9a370c7fe1cb13edd126128cfddc79100852913
- started: 2026-08-20T16:42:02Z
- ended: 2026-08-20T19:06:03Z
- fresh report:
  reports/generated/phase1_protocol_review/protocol_alignment.json
  size 362211 bytes
  mtime 2026-08-20T19:05:58Z
- fresh executed notebook:
  outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb
  size 448911 bytes
  mtime 2026-08-20T19:05:59Z
- environment audit inside the NVIDIA container passed
- nbformat 5.10.4 was available inside the container
- Notebook 03 executed to completion and wrote the report
- the job failed afterward because validate_executed_notebook.py was invoked outside the container and raised ModuleNotFoundError: nbformat

Do not rerun Notebook 03.

Required implementation:

1. Fix scripts/slurm/11_run_phase1_alignment.sbatch so every Python command that depends on project/container packages runs inside the same NVIDIA container through srun:
   - audit_environment.py
   - execute_notebook.py
   - validate_executed_notebook.py
   - review_phase1_results.py

2. Add:
   scripts/slurm/11b_recover_phase1_alignment.sbatch

   This must:
   - require SOURCE_ALIGNMENT_JOB_ID=407;
   - validate the existing report and notebook without executing Notebook 03;
   - verify nonzero sizes and record SHA-256 hashes;
   - verify that the notebook contains completed code cells and no error outputs;
   - run review_phase1_results.py inside the container;
   - preserve the original outputs/private/run_status/407/FAILED.json unchanged;
   - create a new recovery job-specific RUNNING/COMPLETE/FAILED sentinel;
   - record both the original source commit and the recovery-validator commit;
   - mark success only if every postcondition passes;
   - never infer success from file existence alone.

3. Create repository-local context files:
   - AGENTS.md
   - docs/CURRENT_SCIENTIFIC_PROTOCOL.md
   - docs/PROJECT_STATE.md
   - docs/DECISION_LOG.md
   - docs/DGX_BATCH_RUNBOOK.md
   - docs/NEXT_ACTION.md

   These must concisely encode:
   - the pilot panel versus final four-task panel;
   - official prediction-time censoring;
   - 30/90-day prediction-relative windows;
   - exact deduplication;
   - C0–C3 definitions;
   - the final window remaining correct in C1–C3;
   - latest-window/latest-code as separate baselines;
   - M1/M2/M3 definitions;
   - test-set prohibition;
   - governed-data privacy boundaries;
   - current source and DGX state;
   - the next required action.

4. Add a privacy-safe handoff utility:
   scripts/build_handoff_bundle.py

   It may read only:
   - Git metadata;
   - aggregate reports;
   - job sentinels;
   - environment audit summaries;
   - protocol/configuration hashes.

   It must never include raw histories, subject IDs, prediction timestamps, embeddings, labels, predictions, or private manifest rows. Its output should be a concise Markdown or JSON file that can safely be pasted into another Codex task.

5. Update DEV-013 to describe the container-boundary validator defect. Do not mark it resolved until a real recovery job produces COMPLETE.

6. Add focused tests for:
   - containerized validation command construction;
   - recovery rejection of stale/missing outputs;
   - recovery rejection of incomplete/error notebooks;
   - no overwrite of the original failed sentinel;
   - handoff-bundle privacy;
   - atomic final status behavior.

7. Run locally:
   - .venv/bin/python -m pytest
   - .venv/bin/python -m ruff check .
   - bash -n on all changed Slurm scripts
   - bash scripts/git/pre_push_safety.sh

Do not:
- run EHRSHOT;
- run a model;
- access test performance;
- download anything;
- modify patient-level artifacts;
- commit or push.

Finish with:
- a concise root-cause explanation;
- changed files;
- test results;
- the exact git commit suggestion;
- the exact DGX recovery submission command using the batch partition;
- exact completion-verification commands;
- exact aggregate files to copy back to the Mac.
After Codex finishes, review, commit, and push:
git status --short
git diff --check
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
bash scripts/git/pre_push_safety.sh

git add AGENTS.md docs scripts tests
git commit -m "Recover Phase 2 validation and add project handoff"
git push origin main
