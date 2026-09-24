# Project state

Status date: 2026-09-22

## Current structured M2 amendment (supersedes every older recovery below)

- User-provided DGX evidence: job 980 at `0d6ad24cbc86bc865542c96d65c665a0ab8a673e`
  FAILED after checking 32 training inputs; 24 passed. Private evidence hashes were verified on
  DGX. Remaining failures include changed recurrence, unsupported fields/trends, trend retention,
  and length-limited responses. A completed transport response alone is not fidelity evidence.
- Local synthetic inspection also demonstrated defects in old trend/text handling and distinct-
  value grouping (e.g. 1,2,1 becoming 1,2). The historical implementation is preserved, not patched
  into prior caches. Neither all old failures nor all old accepted rows are thereby adjudicated.
- William explicitly authorized a new ordered structured-summary version, with the same source,
  instruction model and 0.6B encoder. New files isolate source transformation, strict JSON/schema
  checks, context-specific fidelity and DGX engineering evidence under `m2_structured_v1`.
- The local implementation schedules only two 32-input train-only engineering jobs, development
  then independent confirmation. Four concurrent requests, two fixed attempts, 10-minute request
  timeout, failure circuit breaker and a 3.5-hour generation deadline bound each 4-hour job.
  Every received response is retained privately; JSON artifacts are atomic and no-clobber.
- No new DGX job was submitted locally. No real data, model, or test performance was run locally.
  Real endpoint schema support, fidelity and runtime remain unverified for this version.
- Production worker/cache integration, capacity sizing, full train/validation generation,
  embeddings, frozen 40-history manual review, and validation fit remain gated work, not completed
  outputs. Do not launch the old M2 graph to consume this version. See
  `docs/M2_STRUCTURED_V1_RUNBOOK.md` for the complete deployment and engineering run instructions.
- Canonical M3 470/471 and report 570 remain complete; M1's one-time test remains closed.

## Historical singleton recovery (superseded by the structured amendment)

- User-reported array 596 inventory: 12 COMPLETE, 369 FAILED, 387 unfinished; 596/597/598 canceled.
  These are job counts, not summary-row counts or independent clinical-error counts.
- First-error log classification before the final cancellation: 333 recurrence, 15 format,
  11 trend, 4 unsupported-fact, 1 association, 1 endpoint/other (365 failed logs).
- Confirmed source defect: grouped source emits `occurrences=1`, output grammar rejected it.
  Repair admits the semantically equivalent explicit singleton, not changed counts; all scientific
  controls and existing cache identities stay unchanged. Existing valid payloads remain reusable.
- New shard execution reports are isolated by source commit to avoid overwriting old reports.
- New DGX-only canary selects 32 training inputs independent of outcome (C1/C3, all four tasks,
  four length strata), records rejected API responses privately, and checks original evidence hashes.
  No full scientific graph is automatically launched. No real model/data ran locally.
- DGX TCP/22 was reachable during implementation, but noninteractive SSH authentication failed.
  No new job has been submitted by this repair. Follow `docs/M2_JOB596_RECOVERY.md` after deployment.
- M1 held-out remains closed and unchanged; canonical M3 remains complete. This is a validation-only
  M2 extension, not a new test-set selection. A fact-ID/extractive redesign has NOT been substituted.

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

- The five-prompt implementation workflow is complete. Prompt 5 retained frozen scientific source
  commit `731545119df7a313883e60967039b5e0ddd2a2f1` and used recovery execution commit
  `e9f8f5901037ddc8f5a6a5ea42019275fae0fd9e`. Jobs 572/573 froze the manifest and built the
  deterministic held-out cohort; the original `574_0`-`574_3` failures remain preserved. Recovery
  preparation jobs `583_0`-`583_3`, embedding jobs `585_0`-`585_3`, fit/report job 586, and
  publication job 587 all wrote COMPLETE and passed the final semantic review.
- The held-out report records `M1_PRIMARY_ONE_TIME_HELD_OUT_COMPLETE_NO_DESIGN_CHANGES_PERMITTED`,
  `test_evaluated_once=true`, `protocol_modified_after_test_access=false`, and no test performance
  in job logs. Its SHA-256 is
  `9c8af6cab8d1ec48e168b6ea6c1f045bf6dc4705f6f7ceccc21a8b7fabb8031f`. The publication sentinel
  SHA-256 is `e0e06c759379e9b9bd08fc78ce21114aef853eb71fe366a8065d3129dfc38c3c`, and the copied handoff
  SHA-256 is `71a490cdf3f6bfe409a56a01ea5b6e21991c27ab05e1a02c7c1454491711e5db`.
- Held-out C1 equal-weight macro-AUROC was 0.646 (95% CI 0.618 to 0.672). The co-primary contrast
  versus latest-window was -0.014 (95% CI -0.032 to 0.003; Holm p=0.224), and the contrast versus
  three-seed shuffled-time probabilities was 0.002 (95% CI -0.001 to 0.006; Holm p=0.224). These
  null/negative results are final. M3 remains validation-only and M2 remains incomplete future work.
- Prompt 4 is closed under the explicit 2026-09-07 scope amendment. M3 reference computation and
  aggregation are complete and semantically valid: jobs 470/471 and report job 570, aggregate
  SHA-256 `7ff518916137317475f8e8a867ea6bcb195cf3c09a27aed67f30b1dc11eb7030`.
  M3 remains validation-only and will not be run on held-out test. M2 did not complete despite
  fail-closed fidelity/runtime attempts; it is future work and no M2 output is eligible for
  selection or held-out evaluation.
- Prompt 5's first preparation graph failed before serialization because the shared
  `prepare_instance` guard did not expose the outer caller's explicit test-only authorization.
  Recovery commit `e9f8f59` added the default-closed boundary flag and separated frozen scientific
  provenance from execution provenance. It resumed the same manifest without refreezing or
  rebuilding the cohort and is now verified complete.

- Jeff requested resumption of M2 after reviewing the completed M1 study. This is an explicitly
  validation-only post-completion extension: it retains the fidelity-v7 summarization contract,
  the on-prem Qwen3.6 endpoint, and the frozen Qwen3-Embedding-0.6B encoder. It cannot access test
  outcomes, revise M1, or trigger M3 recomputation. No resumed M2 job has yet been launched because
  the Mac was off the MedNet route during preparation on 2026-09-14.
- Root cause of historical launcher job 568 is now encoded in source: its Slurm `MaxArraySize`
  probe allowed `awk` to exit before `scontrol` finished while `pipefail` was active, producing
  exit 141 before any fidelity-v7 M2 array was submitted. The repaired probe consumes the full
  stream, and the restart wrapper no longer lets the historical armed-job environment overwrite
  the current source commit. A fresh synthetic capacity ladder is still required before the next
  M2 launch because endpoint capacity is operational evidence and may have changed.

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
- Prompt 4 source is implemented in `configs/experiments/final_m2_m3.yaml`,
  `src/temporal_ehr/final_m23/`, notebooks 08-09, and Slurm jobs 24-28. M2 is a non-recursive,
  deterministic-where-supported, on-premise-only summary path with cache identity over source hash,
  condition, prompt, served-model revision, and decoding configuration. Its validation-only
  fidelity sample is frozen at 40 histories, five per task crossed with correct/shuffled time. M3
  reconstructs complete locked windows and verifies them against Prompt 3's patient-window hashes,
  then uses frozen Qwen3-Embedding-0.6B at 512 tokens per window, at most 64 recent nonempty windows, a
  mean-pool logistic comparator, a fixed 256-dimensional projection, a one-layer unidirectional GRU
  with hidden size 128/dropout 0.1/AdamW, validation-only early stopping, and three frozen seeds.
- The source implements C1 reference execution first, the exact 0.005 validation macro-AUROC
  approach-selection rule, conditional selected-approach sensitivities, and a cryptographically
  hashed freeze manifest. Local validation passed 118 synthetic tests, Ruff, shell syntax, context,
  and privacy checks without running real data, summaries, embeddings, GRUs, or test evaluation.
- A pre-execution alignment review found that the initial Prompt 4 source used validation log loss
  rather than the August protocol's validation AUROC for M3 early stopping, automated but did not
  human-gate the required 40-history M2 review, and omitted latest-window from the frozen
  selected-approach co-primary authorization. The working update restores AUROC early stopping,
  adds a private manual-review packet/decision/aggregate gate, and requires B1 plus all three C3
  seeds for a selected M2/M3 approach. M2 is explicitly defined as compression of the same
  content-matched 8,192-token source evidence as M1, pending Jeff confirmation.
- Engineering acceleration is limited to outcome-independent execution: the GPU-free M2 summary
  array runs two tasks concurrently with four requests each to the separate on-premise vLLM server;
  M2 embeddings run in a dependent GPU array; M3 uses frozen-window embedding batch size 32; and
  truthful Slurm reservations replace blanket 110G requests. A new synthetic-only capacity probe
  must confirm throughput and memory before governed reference work.
- M3 scheduling now has a seed-independent four-task preparation array followed by its 12
  task/seed training array. This removes eight redundant frozen-encoder loads in the reference
  stage, makes every seed consume one hashed preparation report, permits two lightweight GRU jobs
  concurrently by default, and projects retained windows before padding to reduce peak memory.
  A read-only status helper reports live Slurm state plus COMPLETE/FAILED sentinels without opening
  governed artifacts. No Prompt 4 governed job has been declared complete from these source changes.
- Before this alignment/scheduling update reached `main`, the old `a28eab1` Prompt 4 helper
  submitted M2 array 456, M2 fit 458, M3 array 459, and M3 fit 460. At 2026-09-02T01:00Z, 456_0
  was FAILED, 456_1 was consuming the old 110G/GPU reservation, and the M3 array was blocked for
  resources. This graph is superseded and must be canceled/preserved. New M2 embedding paths include
  the full source commit, while valid summary rows remain reusable only through their immutable
  source/condition/prompt/model/decoding key.
- After aligned commit `bd4ae35` passed its in-container hard gate, M3 preparation tasks 465_0
  through 465_3 failed before scientific Python because `26a_m3_prepare_task_array.sbatch` did not
  define the `CONTAINER_IMAGE` variable required by `final_job_common.sh`; dependent seed array 469
  became `DependencyNeverSatisfied`. No source reconstruction, embedding, fitting, or result ran.
  The pending hotfix defines the image and tests every common-wrapper Slurm script for that contract.
- At commit `1a97d8f`, replacement M3 preparation 470_0-470_3 and seed training 471_0-471_11
  completed with exit-0 sentinels and clean logs. These are the canonical M3 reference compute.
  M2 job 502_0 was canceled after a privacy-safe audit found 1,286 null-like summaries among 1,294
  cached rows. The endpoint itself passed the exact synthetic prompt when
  `chat_template_kwargs.enable_thinking=false`; the defect was the client converting JSON null to
  the literal string `None`. Job 502_0 has only its preserved RUNNING sentinel. The recovery source
  makes non-thinking part of the decoding/cache hash, rejects non-text and null-like content, adds
  a live synthetic preflight, and reuses verified M3 jobs 470/471 without rerunning them.
- Commit `08a639b` supplied that non-thinking recovery. M2 job 507_1 completed all 2,533
  readmission reference summaries with non-null content, but a governed validation-only spot check
  exposed an output-limit failure mode: one summary enumerated a repeated measurement until its
  ending, omitted other source facts, and added code-name interpretations absent from the source.
  The prior automated audit also expected bracketed concepts although the serializer emits plain
  clinical codes. Task-0 recovery 513 reused zero rows and generated only 500 of 4,088 after 3h37m,
  making completion within 12 hours infeasible. Cancellation was requested for arrays 507 and 513
  on 2026-09-03; their evidence remains superseded and must be preserved.
- The pending fidelity-v2 source versions the M2 prompt/cache, requires compact structured facts
  without code expansion, rejects every completion whose API finish reason is not `stop`, records
  token counts, removes the vLLM process-start timestamp from the stable served-model identity, and
  repairs fidelity extraction for plain clinical codes and explicit values/units/recurrence. A
  dense synthetic live preflight must pass before any new M2 jobs are submitted. M1, canonical M3
  jobs 470/471, and the held-out test lock are unchanged.
- Commit `f25080d` was pulled to the DGX, but its synthetic-only preflight failed before `sbatch`, so
  it created no recovery report and submitted no governed jobs. Repeated data-free probes isolated
  the cause: Qwen sometimes reported 65 occurrences for 64 identical input facts despite
  temperature zero and the fixed seed. The pending narrow repair deterministically collapses exact
  duplicate facts and supplies the count before inference, versions that transform in the summary
  cache, and requires three consecutive dense synthetic passes. Five local-to-on-premise probes of
  the revised input/prompt all passed; governed execution remains pending.
- Commit `a316c98` then implemented exact-repeat precompaction and submitted M2 reference array 519.
  It did not establish M2 completion: tasks 519_0 and 519_1 reached the 12-hour limit, 519_2 was
  canceled, and 519_3 failed after rejecting 549 of 650 length-truncated responses. Dependent M2
  embedding 520 and review 521 were canceled. The preserved current-version successful-row counts
  are 396 anemia, 50 readmission, 67 hypertension, and 101 CheXpert; they are superseded engineering
  evidence and cannot satisfy the next prompt/cache identity.
- The fidelity-v3 recovery design grouped every window by exact code/source/unit before Qwen,
  preserves distinct numeric/text values in first-occurrence order and exact occurrence totals,
  raises the completion ceiling from 2,048 to 4,096, requires `finish_reason=stop`, and validates
  every accepted summary against the frozen 8,192-token embedding tokenizer. It records progress
  every 25 new rows, preserves all earlier attempts, reruns only M2, and reuses M3 jobs 470/471.
- Commit `818103e` submitted fidelity-v3 M2 array 529. Tasks 529_0 and 529_1 reached 12 hours,
  529_2 was intentionally canceled after 7h35m when the graph was already unsatisfiable, and
  529_3 failed after 333 CheXpert responses reached `finish_reason=length` at 4,096 tokens. All
  four have preserved FAILED sentinels; downstream 530/531 were canceled. The last aggregate-only
  cache inventory showed 677 anemia, 881 readmission, 640-plus hypertension, and 317 CheXpert
  fidelity-v3 candidate rows. These are candidates, not completion evidence.
- The fidelity-v4 recovery design retained the exact primary decoding hash, reused only candidate
  rows whose content is an exact reconstructed grouped-source copy and passes the frozen 8,192-token
  tokenizer, and generates only missing/rejected rows under a separate 8,192-token overflow hash.
  Work is partitioned into eight deterministic shards per task with at most two active and four
  requests per shard. M3 470/471 remains canonical; test performance remains locked.
- Commit `8682cb4` submitted fidelity-v4 array 536. Shards 536_0/536_1 saved 400/375 new exact rows
  but failed closed after 49/34 other responses ended with `finish_reason=length` at the 8,192-token
  generation ceiling. This demonstrates a tokenizer-length mismatch rather than endpoint or Slurm
  failure. Downstream 537/538 were canceled and all cache/status evidence is preserved.
- Fidelity-v5 adds only a separately hashed 16,384-token instruction-generation tier for unresolved
  rows. Previously accepted 4,096/8,192 rows still require reconstructed exact equality and are
  reused; the frozen downstream embedding limit remains 8,192 tokens. M3 remains unchanged and test
  performance remains locked.
- Commit `223acb0` submitted fidelity-v5 array 545. Shards 545_0-545_7 all failed before the
  12-hour limit: six contained grouped exact copies exceeding the frozen 8,192-token downstream
  embedding context, and two contained assistant text that was not an exact copy. Shards 545_8 and
  545_9 plus downstream jobs 548/549 were canceled after the graph became unsatisfiable. M3 470/471
  was reverified COMPLETE and remains canonical. No M2 fit, selection, or held-out evaluation ran.
- Fidelity-v6 replaces the contradictory exact-copy rule with the bounded structured summary
  specified by Prompt 4. It uses a new prompt/cache/validation identity, one 6,144-token decoding
  configuration, strict row-level unsupported-fact and timing guards, the existing fidelity floors,
  and the unchanged 8,192-token embedding limit. Work is split into 16 shards per task, four active
  with two requests each; total endpoint concurrency remains eight while each Slurm job becomes
  smaller. Earlier cache rows and job evidence are preserved but cannot satisfy fidelity-v6 keys.
  The full 162-test local synthetic suite, tracked-source Ruff checks, changed-shell syntax checks,
  durable-context check, and pre-push privacy/safety check pass; governed DGX validation is pending.
- Fidelity-v6 array 559 at commit `38e2b7b8` demonstrated valid resumable row admission but an
  execution rate incompatible with its 12-hour shards. At approximately 11.6 hours, its first four
  anemia shards had accepted 25-50 rows each at 3.4-5.5 rows/hour/shard and projected 37-69
  additional hours. Jobs 560/561 remained dependency-blocked. The fidelity-v7 recovery preserves
  every accepted fidelity-v6 row and the complete scientific/cache identity, measures 8/16/24/32
  endpoint concurrency with synthetic records only, selects concurrency using throughput/failure
  evidence only, and uses 192 deterministic shards per task with eight requests per active shard.
  The unattended launcher advances from capacity measurement into M2 only after exact COMPLETE
  evidence and separately aggregates existing M3 reference predictions while M2 runs; it never
  reruns M3 task/seed computation or performs approach selection early. M3 470/471 and the held-out
  test lock remain unchanged.
