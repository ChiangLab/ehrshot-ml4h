# Project state

Status date: 2026-08-22

## Source state

- Baseline source commit for DGX job 407:
  `f9a370c7fe1cb13edd126128cfddc79100852913`.
- That commit contains the 12-hour Phase 1 alignment runner and protocol-aligned serializer QC.
- Recovery-validator commit `e12d475d5ec2bc1f6c02aeda28e1a252ac84d6ea` is on `main`.
- The pilot cohort remains the three-task engineering panel. The separate final four-task panel is
  now prospectively locked before pilot embedding results.
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

- Do not rerun Notebook 03. Recovery job 408 completed on 2026-08-22 with sentinel exit status 0,
  passed report/notebook semantics, recorded report SHA-256
  `38ec5258ae89f4ec1935c7969873681eb1b155f55521f92bce2ba7b4784fc681`, and preserved job 407's
  FAILED sentinel. Slurm accounting is disabled, so `sacct` history is not available.
- The validated report covers 9,300 train/validation pilot instances and passes content matching;
  no model or test performance was run. DEV-013 is resolved by job-408 evidence.

## Current build state

- Prompt 2 authorizes the isolated complete three-task engineering pilot.
- This authorization supersedes the older sequencing proposals in checked-in prompts 03 and 04;
  they remain historical records and are not the current execution plan.
- Final-panel cohort construction, M2, M3, Qwen 8B, and held-out evaluation remain out of scope.
