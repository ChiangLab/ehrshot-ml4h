# Project state

Status date: 2026-09-01

## Source state

- Prompt 3 source and its fit-recovery validator are on Mac, `origin/main`, and the DGX through
  commit `ef1b0debfa963ee365645ec2f754d8d129e071ce`.
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
  verified-node hotfix was then committed as `f45e61c` to pin the fresh Prompt 3 chain to
  `dgit00038673`. M2, M3, and held-out evaluation remain out of scope.
- Commit `f45e61c0861b246a0f0061f7dcf4e9aa65ced880` pinned the chain to the verified node. In the
  resulting jobs, synthetic 8B smoke 429 passed and foundation 430 reached governed cohort
  construction, then failed closed because an official anemia patient/time group had conflicting
  binary targets. Jobs 431–433 were canceled. An aggregate-only governed audit found 183 duplicate
  time groups in anemia and 752 in CheXpert across train/validation, none in readmission or
  hypertension, and only a suppressed small number of patients losing all candidates. The pending
  repair excludes every duplicated patient/task/time group before ranking without consulting label
  values, then reports privacy-suppressed aggregate accounting. It passed 93 local synthetic tests,
  Ruff, shell syntax, context, privacy, and pre-push checks. No final model or test result ran.
- At commit `1eb0d35`, synthetic 8B smoke job 436 passed and foundation job 437 completed in 3h59m41s.
  Its hard gates, four-task foundation command, semantic reviewer, aggregate notebook, notebook
  validator, immutable cohort, and COMPLETE/exit-0 sentinel all passed. Throughput job 438 loaded
  both models, then correctly failed its runtime gate: measured 0.6B speed was 1.40777 documents/s,
  and anemia's 30,056 rows project to 42,700.11 seconds under the required 2x contingency. The
  43,200-second job limit leaves 499.89 seconds; only the additional 900-second reserve caused the
  failure. Jobs 439–440 were canceled and no embedding/model result ran.
- The pending recovery retains 2x contingency, changes only the extra reserve from 900 to 300
  seconds, validates exact reuse of job 437 across that execution-only change, and resubmits only
  throughput plus downstream work. No cohort, prepared input, task, model, condition, or outcome
  definition changes.
- Commit `77c47f322b98fc74196cc56a0a16edc43604c887` implemented that exact reserve recovery.
  Throughput job 444 completed, and embedding array tasks 445_0 through 445_3 each wrote
  COMPLETE/exit-0 at that commit. Their four aggregate embedding reports and immutable verified
  caches exist; do not rerun or re-encode them.
- Fit job 446 failed before reading caches or fitting a classifier because `report.py` still
  required the reused job-437 foundation's source commit to equal the current fit commit. Recovery
  job 451 then validated exact jobs 437/444/445 and the unchanged failed-job-446 hash, fit the
  validation-only readout, executed and validated Notebook 06, and wrote COMPLETE/exit 0. The final
  M1 report is `reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json`, SHA-256
  `6d16d6b2494042fd2c84d25ef790f030355738cf8512fb87cf5477f73a3e0c7a`. The Prompt 3 handoff SHA-256
  is `2bd2223b13b9395ee7a725da9c260f1560c387db6cf5177f28f35999e0c56814`; it records jobs 437, 444,
  445_0 through 445_3, 446, and 451 without patient-level content. Prompt 3 is COMPLETE, M1 is
  frozen as reported, and held-out test performance remains untouched. Prompt 4 M2/M3 source is the
  active phase.
