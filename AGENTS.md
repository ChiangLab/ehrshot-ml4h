# Repository operating context

Active override after verified jobs 988/989/990 (2026-09-29): structured-v2 engineering PASSED
(64 regression and 32 fresh confirmation inputs). Use `docs/M2_STRUCTURED_V2_PRODUCTION.md`.
The separate reference production implementation is locally testable; real production remains
pending. Preserve the engineering commit `6c80719266485c861fd8aa13f931eebcf5bfc8f8`, all prior
failures and scientific identities. Freeze worklists, reuse only revalidated exact-v2 generations,
run bounded chunks, gate embeddings on complete coverage, and require the frozen 40-history
human review before validation fitting. Do not call reference completion the full sensitivity
matrix. Do not invoke old selection/freeze/test launchers. M1/M3 and test remain closed.

The optional governed project explorer in `docs/PROJECT_EXPLORER.md` is a review interface, not a
new experiment. It may index only train/validation cohort rows and pre-T source events, M1-M3
development artifacts, and non-test sentinels. It must bind to DGX localhost and remain under
`outputs/private`; the one-time held-out artifacts remain outside the catalog. The protocol defines
no M4. Full embedding arrays remain immutable and are indexed rather than copied into DuckDB.

Active override (2026-09-25; supersedes historical recovery instructions below): replay 985
passed semantic verification; confirmation 986 has 29/32 PASS and genuine duplicate/changed/
reordered-series rejections. Preserve both, including 986 FAILED. Use
`docs/M2_STRUCTURED_V2_AFTER_986.md`. The authorized v2 prompt repair retains the clinical source,
schema, decoding, instruction model, 0.6B encoder, compact renderer and strict fidelity gates.
JSON object keys are serialized canonically; clinical arrays are never sorted. Version all new
generation keys. Regression covers all 64 previously inspected inputs; a fresh 32-input training
confirmation excludes all 980/982/986 engineering people and is frozen before new generation.
The launcher queues regression -> confirmation -> readiness only. A failed gate blocks successors.
Production integration, full generation, embeddings, frozen manual review and validation fitting
remain outstanding; readiness is NOT scientific M2 completion. No M1/M3/test rerun is authorized.

Current override (2026-09-24): development 982 at `f9cce3d` checked 32 inputs, with 30 accepted;
the user-provided hash-verified replay found all 32 latest responses factually valid with full
measured retention. Two v1 rendered texts were 8,248/8,265 tokens, exceeding 8,192. Confirmation
983 did not start. Follow `docs/M2_RENDER_V2_RECOVERY.md`: replay saved 982 responses with the
reversible compact renderer, then run the original frozen independent confirmation sample.
Generation prompt/schema/model/source/decoding remain pinned to f9cce3d; embedding text and its
cache key are versioned separately. No facts, counts, order, windows, or retention floors are
changed. The same 0.6B tokenizer enforces 8,192 including the legend. Preserve 982 FAILED and all
earlier evidence. Real render-v2 token fit remains pending. Do not resubmit the old development
launcher or any full M2 array. No M1/M3/test rerun is authorized.

Current M2 recovery override (2026-09-22): job 980 failed its 32-input canary (24 PASS).
William authorized a separately versioned single-pass structured summary preserving ordered
measurements. Follow `docs/M2_STRUCTURED_V1_RUNBOOK.md`, not historical restart commands below.
The new path is isolated from all legacy caches and validators. It uses the same M1-retained
clinical source, on-premise Qwen instruction model, 0.6B encoder, and train/validation boundary.
This is an explicit source-format/prompt/schema amendment, not merely a scheduling fix.
Only two gated 32-input training-only engineering jobs are implemented for launch; no full
scientific graph is armed. Independent confirmation excludes all development/job-980 people.
Do not re-arm 596 or relabel 980 as COMPLETE. Require real semantic/hash-verified evidence before
production integration; retain the frozen 40-history manual fidelity gate. M1 and M3 stay frozen;
the already-completed one-time test is closed and must not be reopened.

Before changing scientific or DGX execution code, read these files in order:

1. `docs/CURRENT_SCIENTIFIC_PROTOCOL.md`
2. `docs/PROJECT_STATE.md`
3. `docs/DECISION_LOG.md`
4. `docs/NEXT_ACTION.md`
5. `docs/DGX_BATCH_RUNBOOK.md` when Slurm or DGX work is involved
6. `docs/PROTOCOL_DEVIATIONS.md` and `docs/SERIALIZATION_PROTOCOL.md` for detailed contracts

Treat `docs/NEXT_ACTION.md` as the current handoff, but verify it against Git and job sentinels.
Never infer scientific success from file existence or a partial notebook. Require scheduler state,
exit status, an atomic COMPLETE sentinel, semantic validators, and recorded hashes.

Do not open raw EHRSHOT rows or patient-level artifacts for code assistance. Never print or commit
subject identifiers, prediction timestamps, serialized histories, labels, predictions, embeddings,
or private manifest rows. Governed work stays on MedNet/DGX; off-VPN work uses source code,
synthetic fixtures, and reviewed aggregate reports only. Do not access held-out test performance
until the protocol and test gate are explicitly frozen.

Do not silently lock the proposed panel, rerun an expensive job, run a model, or overwrite a job
sentinel. Preserve source notebooks and write executed copies only under `outputs/private/`.

Current critical state: recovery job 408 verified job 407's completed Notebook 03 and aggregate
without rerunning it. The complete Prompt 2 engineering pilot then passed at commit
`ae35c50982ed56cee9a90875cdc98a05f63ffd54`: preparation 415, throughput 416, embedding array
417_0 through 417_2, and fit/report 418 all wrote COMPLETE with exit status 0. The pilot aggregate
SHA-256 is `6b2d812cf430ed186fae00615e3b302dd8da58294b23902a74ef514c4047494d` and records
`PILOT_COMPLETE_FINAL_PANEL_UNCHANGED`. Historical `sacct` is unavailable, so durable sentinels,
clean logs, hashes, and semantic validation are the retained evidence.

The final four-task panel is prospectively LOCKED and remains isolated from the completed
three-task engineering PILOT. A pilot cohort or cache must never satisfy a final-panel gate. Prompt
3 is COMPLETE: foundation 437, throughput 444, task embeddings 445_0 through 445_3, and fit-only
recovery 451 all have validated COMPLETE/exit-0 evidence. Failed fit 446 remains preserved. Job 451
ran at commit `ef1b0debfa963ee365645ec2f754d8d129e071ce`; the final M1 aggregate SHA-256 is
`6d16d6b2494042fd2c84d25ef790f030355738cf8512fb87cf5477f73a3e0c7a` and the privacy-safe Prompt 3
handoff SHA-256 is `2bd2223b13b9395ee7a725da9c260f1560c387db6cf5177f28f35999e0c56814`.
The report records `FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED`; no M2, M3, or held-out
evaluation was performed.

Prompt 4 source is implemented for M2 one-pass summaries and M3 per-window sequences. M2 is
hard-bound to the on-premise `10.45.177.17:8000` endpoint, uses immutable source/condition/prompt/
model-revision/decoding cache identity, and includes the prespecified balanced 40-history fidelity
audit plus a required private manual review of the same sample. M2 uses the same content-matched
8,192-token source evidence as M1 pending Jeff confirmation. M3 uses frozen 512-token Qwen window
embeddings, at most 64 recent nonempty windows, a
mean-pool logistic comparator, and a one-layer 128-hidden-unit unidirectional GRU under seeds
20260815-20260817 with validation-AUROC early stopping. The reference arrays must complete, the
manual M2 review must be finalized, and `approach_selection.json` must pass before conditional
sensitivity. The selected approach must have B1 latest-window and all three C3 seeds before freeze.
Execution separates M3's seed-independent source/embedding preparation from its 12 seed-training
jobs. The frozen encoder is loaded once per task/stage rather than once per seed, and two small GRU
jobs may share the DGX after preparation; this is scheduling only and changes no model cell.
An obsolete Prompt 4 graph was submitted on the DGX from pre-alignment commit `a28eab1`: M2 array
456, M2 fit 458, M3 array 459, and M3 fit 460. Task 456_0 failed and the remaining graph must be
canceled and preserved, never treated as final evidence. Immutable per-row summary records may be
reused by their scientific cache key, but new embedding caches are isolated by full source commit.
The first split-M3 submission at aligned commit `bd4ae35` also produced no scientific work:
preparation array 465 failed before container startup because script 26a omitted `CONTAINER_IMAGE`,
and seed array 469 was dependency-blocked. Preserve those failures and use the tested wrapper hotfix
before submitting replacement M3 preparation/training IDs.
The corrected M3 reference compute is now complete at commit `1a97d8f`: preparation jobs 470_0
through 470_3 and seed jobs 471_0 through 471_11 all have COMPLETE/exit-0 evidence. Preserve and
reuse that exact M3 evidence. M2 summary job 502_0 was canceled after an aggregate-only audit found
that most cached assistant contents were JSON null converted to literal `None`; it ended with only
its original RUNNING sentinel. The first non-thinking recovery at commit `08a639b` produced valid
non-null rows, including COMPLETE job 507_1, but governed validation inspection found a long summary
that spent its output budget enumerating one repeated value, omitted other supplied facts, and added
unsupported code names. Array 507 and task-0 recovery 513 were canceled on 2026-09-03; preserve all
their logs, sentinels, and caches. The pending fidelity-v2 recovery requires compact structured
output, exact code copying without expansion, `finish_reason=stop`, token metadata, plain-code-aware
fidelity parsing, a restart-stable served-model identity, and a dense synthetic preflight. Its new
prompt hash prevents every 502/507/513 row from satisfying the recovery cache.
Commit `f25080d` reached the synthetic-only DGX preflight and submitted no jobs because Qwen
occasionally counted 64 identical synthetic facts as 65. Do not retry until the pending exact-repeat
precompaction hotfix is committed: Python must collapse exact duplicate facts within each window,
write the true `occurrences=N` before inference, and bind that transform into a new cache schema.
Do not run either real path locally or submit sensitivity before these gates pass.

Commit `a316c98` implemented exact-repeat precompaction, but M2 array 519 did not establish valid
reference completion: tasks 0/1 reached the 12-hour limit, task 2 was canceled, and task 3 rejected
549 of 650 completions because `finish_reason=length`. Downstream jobs 520/521 were canceled.
Preserve every 519-521 status, log, and cache. The active recovery is fidelity-v3: group facts
deterministically within each unchanged window by code/source/unit, preserve distinct numeric/text
values in first-occurrence order and exact total occurrence counts, use a new prompt/compaction/cache
identity, reject non-`stop` or over-8,192-token summaries, and rerun only M2 while reusing M3 470/471.

Fidelity-v3 array 529 at commit `818103e` preserved useful cache rows but did not complete: tasks
0-2 ended with exit 143 at the site limit or explicit cancellation, and task 3 rejected 333
CheXpert responses with `finish_reason=length` at the 4,096-token ceiling. Jobs 530/531 were
canceled and all `529_*` sentinels/caches are immutable evidence. The active fidelity-v4 recovery
keeps the primary 4,096-token cache identity unchanged, reuses a row only after exact grouped-source
copy and downstream-token validation, writes missing rows under a separate 8,192-token decoding
identity, and assigns work to eight deterministic shards per task with at most two active shards.
Do not run M2 from a laptop, rerun M3 470/471, delete prior caches, or access test performance.

Fidelity-v4 array 536 at commit `8682cb4` proved that the instruction model's generation-token
count can exceed 8,192 even when the exact grouped output passes the separate frozen 8,192-token
embedding tokenizer. Shards 536_0/536_1 failed closed on 49/34 `finish_reason=length` responses
while preserving 400/375 successful new rows. Fidelity-v5 added a 16,384-token generation tier, but
job 545 then proved exact copying incompatible with bounded summarization: shards 545_0-545_7 all
failed, six on over-context exact outputs and two on non-exact copies; 545_8/545_9 and downstream
548/549 were canceled. Preserve 536-538 and 545/548/549; do not rerun those recovery versions.

Fidelity-v6 uses one new 6,144-token bounded structured-summary identity,
requires `finish_reason=stop`, rejects unsupported codes/values/units, altered recurrence, invalid
format, changed time-label sequence, or movement of a fact between time blocks, and reapplies
row-level retention plus the unchanged 8,192-
token downstream gate before cache admission. It uses 16 shards per task, four active shards, two
requests per shard, and three fixed contract attempts. This restores actual compression required by
Prompt 4 without changing M1 source evidence, M3 470/471, any outcome definition, or the test lock.
Array 559 at commit `38e2b7b8` proved that identity can admit valid rows, but the first four shards
ran at only 3.4-5.5 accepted rows/hour/shard and projected 37-69 hours remaining near the 12-hour
limit. Preserve and stop jobs 559-561. The active fidelity-v7 recovery keeps the exact fidelity-v6
scientific/cache identity, revalidates every candidate row, runs a synthetic-only 8/16/24/32
capacity ladder, and changes only execution to eight requests per shard and 192 shards per task.
Do not combine any legacy prompt identity, rerun M3 470/471, or access held-out performance.
Use `scripts/slurm/arm_prompt4_m2_overnight_v7.sh` for unattended execution: it launches M2 only
after same-source capacity COMPLETE evidence and independently aggregates the existing M3 reference
predictions without rerunning M3 computation or performing approach selection before M2 finishes.

On 2026-09-07 William explicitly authorized a deadline-preserving scope amendment after the
governed M2 capacity/launcher attempt failed before launching M2. M1 direct serialization is the
sole primary approach for the one-time held-out evaluation. The completed M3 window-sequence model
is retained as validation-only secondary evidence; it is not run on held-out test. M2 one-pass
summary embedding remains incomplete future work and is neither selected nor a held-out gate. This
amendment does not change the locked tasks, cohorts, prediction times, temporal conditions,
baselines, M1 hyperparameters, validation-selected penalties, or statistical plan. The immutable
M1-only manifest was frozen at commit `7315451`; freeze 572 and deterministic cohort 573 are
COMPLETE, so the private access audit is open. No held-out embedding, prediction, or performance
was produced before preparation jobs `574_0`-`574_3` failed on a default-closed shared serializer
guard. Preserve that manifest, cohort, access audit, and every failed sentinel. Recover only the
same manifest using the versioned execution repair, retaining `7315451` as the scientific source
and recording the repair commit separately.

Prompt 5 is COMPLETE. Frozen scientific source commit `7315451` and recovery execution commit
`e9f8f59` produced COMPLETE jobs 583/585/586/587 under the original immutable manifest. The
held-out aggregate SHA-256 is `9c8af6cab8d1ec48e168b6ea6c1f045bf6dc4705f6f7ceccc21a8b7fabb8031f`
and the publication sentinel SHA-256 is
`e0e06c759379e9b9bd08fc78ce21114aef853eb71fe366a8065d3129dfc38c3c`. The report records one-time
evaluation, no post-access protocol modification, and no held-out performance in job logs. Do not
rerun, refit, select a different approach, or alter the analysis in response to these results.

The completed Prompt 2 authorization supersedes the sequencing and narrower scope proposed in
`prompts/03_BUILD_PHASE3_BASELINES.md` and `prompts/04_BUILD_THROUGHPUT_AND_M1_PILOT.md`. Keep those
files as historical planning records. All five canonical implementation prompts are complete.
No active implementation prompt remains; paper finalization must use the frozen aggregate artifacts
without initiating additional held-out computation.

Canonical phase prompts live under `prompts/completed/`, `prompts/next/`, and `prompts/planned/`.
Files numbered 03 and 04 at the top level predate the current five-prompt execution plan and remain
historical only. `docs/NEXT_ACTION.md` names the current finalization action. After each phase,
update `PROJECT_STATE.md`, `DECISION_LOG.md` or `PROTOCOL_DEVIATIONS.md` as applicable,
`RESULTS_REGISTRY.md`, and `NEXT_ACTION.md`; run `scripts/check_project_context.py`; generate a
privacy-safe handoff; then move the finished
canonical prompt from `prompts/next/` to `prompts/completed/` and promote exactly one planned prompt
when a successor exists. After Prompt 5, both `prompts/next/` and `prompts/planned/` remain empty.
