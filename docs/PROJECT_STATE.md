# Project state

Status date: 2026-08-21

## Source state

- Baseline source commit for DGX job 407:
  `f9a370c7fe1cb13edd126128cfddc79100852913`.
- That commit contains the 12-hour Phase 1 alignment runner and protocol-aligned serializer QC.
- The current local change set repairs the container boundary and adds an evidence-bound recovery;
  its recovery-validator commit is intentionally not known until the user reviews and commits it.
- The pilot cohort is still the three-task panel. The final four-task panel is proposed, not locked.
- No embedding experiment, local-LLM summarization, model training, or held-out evaluation has run
  as part of this recovery work.

## DGX job 407 evidence

- Job interval: 2026-08-20T16:42:02Z through 2026-08-20T19:06:03Z.
- The environment audit and Notebook 03 execution ran inside the NVIDIA container.
- Fresh aggregate report: `reports/generated/phase1_protocol_review/protocol_alignment.json`,
  362211 bytes, mtime 2026-08-20T19:05:58Z.
- Fresh executed notebook:
  `outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb`, 448911 bytes, mtime
  2026-08-20T19:05:59Z.
- Notebook 03 completed and wrote the report. The subsequent host-side validator failed with
  `ModuleNotFoundError: nbformat`, so job 407 correctly retains `FAILED.json` and is not scientific
  completion evidence by itself.

## Recovery state

- Do not rerun Notebook 03.
- `scripts/slurm/11b_recover_phase1_alignment.sbatch` validates the existing artifacts against the
  exact evidence above, checks all code cells and error outputs, records hashes, runs the aggregate
  reviewer inside the container, preserves job 407's failed sentinel, and writes a separate
  recovery job status.
- DEV-013 remains open until a real recovery job has scheduler state COMPLETED, exit `0:0`, an
  error-free log, and a valid job-specific COMPLETE sentinel.
