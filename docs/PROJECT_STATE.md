# Project state

Status date: 2026-08-29

## Source state

- Prompt 3 source commit `6a0d0accddb0098f713796ca65750ca162a7daeb` is on Mac, `origin/main`,
  and the DGX. The containerized-preflight hotfix is intentionally uncommitted pending review.
- Baseline source commit for DGX job 407:
  `f9a370c7fe1cb13edd126128cfddc79100852913`.
- That commit contains the 12-hour Phase 1 alignment runner and protocol-aligned serializer QC.
- Recovery-validator commit `e12d475d5ec2bc1f6c02aeda28e1a252ac84d6ea` is on `main`.
- The pilot cohort remains the three-task engineering panel. The separate final four-task panel is
  now prospectively locked before pilot embedding results.
- The later Prompt 2 engineering pilot ran embeddings and validation-only readouts independently of
  the earlier recovery work. No local-LLM summarization, final-panel modeling, or held-out
  evaluation has run.

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

- Prompt 2's isolated complete three-task engineering pilot is COMPLETE.
- The pinned Qwen3-Embedding-0.6B artifact is complete on the DGX. Synthetic smoke job 411 loaded
  it but failed the unit-norm contract because SentenceTransformers normalized finite vectors in
  BF16. A corrected synthetic smoke run produced finite 1,024-dimensional embeddings with norms
  from 0.9999999992 to 1.0000000016, verifying explicit FP32 L2 normalization.
- Commit `cb07cb5` contains the FP32 normalization repair and is on `origin/main`. Canonical prompts
  1-5 and the dependency-locked workflow are in commit
  `ae35c50982ed56cee9a90875cdc98a05f63ffd54`, which governed Prompt 2 execution.
- Jobs 415, 416, 417_0, 417_1, 417_2, and 418 all wrote COMPLETE with exit status 0. The in-container
  notebook and aggregate reviewers passed. The copied pilot report has SHA-256
  `6b2d812cf430ed186fae00615e3b302dd8da58294b23902a74ef514c4047494d`; its privacy-safe handoff
  has SHA-256 `e95300137aaa317a65138f9fa55a4f121ab466fb5ed3d869c5f6660e21930042`.
- The historical Phase 2 alignment report remains `NO_GO` because it predates the panel lock. Do
  not rewrite it; the subsequently locked configuration and completed pilot are the later gate
  evidence.
- This authorization supersedes the older sequencing proposals in checked-in prompts 03 and 04;
  they remain historical records and are not the current execution plan.
- Prompt 3 final-panel foundation and M1 source is implemented at commit `6a0d0acc`. It includes
  the exact final cohort selection rule, strict-censored/deduplicated
  patient-window foundation, outcome/shortcut audits, 8,192-token 0.6B embeddings, runtime-only
  8B confirmatory gate, condition-level resumable caches, task array jobs, paired patient
  bootstrap reporting, semantic validators, aggregate notebooks, and privacy-safe handoff support.
- Local validation passed 92 synthetic tests, Ruff, shell syntax, context, and privacy gates. No real
  EHRSHOT row, model, final metric, M2/M3 path, or held-out test performance was accessed locally.
- The complete 15 GB 8B snapshot was transferred from the Mac after DGX download job 421 stalled
  and was canceled. Before scientific submission, the helper's host-side hard-gate CLI failed
  because NumPy exists in the NVIDIA container rather than the DGX host environment; no job was
  created by that attempt. Do not install or override system Python.
- Commit `fe30dda5ed9537633fdff16dfb5ced0f562d41b7` fixed the preflight boundary. Synthetic 8B smoke
  job 423 then passed on `dgit00038673` with local-only 8,192-token inference, 4,096-dimensional
  finite FP32-normalized embeddings, and no patient data. Foundation job 424 was scheduled on
  `dgit00038674` and failed before scientific execution because Pyxis could not create
  `/run/user/2051`; downstream jobs 425–427 were canceled and no final aggregate was created. A
  verified-node hotfix now pins the fresh Prompt 3 chain to `dgit00038673` pending commit/pull/run.
  M2, M3, and held-out evaluation remain out of scope.
