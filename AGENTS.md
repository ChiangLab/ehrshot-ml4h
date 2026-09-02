# Repository operating context

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
its original RUNNING sentinel. The active recovery disables Qwen thinking through a hashed decoding
control, rejects non-text/empty/null-like responses, isolates old cache rows, and reruns M2 only.
Do not run either real path locally or submit sensitivity before these gates pass.

The completed Prompt 2 authorization supersedes the sequencing and narrower scope proposed in
`prompts/03_BUILD_PHASE3_BASELINES.md` and `prompts/04_BUILD_THROUGHPUT_AND_M1_PILOT.md`. Keep those
files as historical planning records. The sole active implementation prompt is
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`. Prompt 4 may use only train/validation evidence and
authorized on-premise summarization; held-out evaluation remains prohibited.

Canonical phase prompts live under `prompts/completed/`, `prompts/next/`, and `prompts/planned/`.
Files numbered 03 and 04 at the top level predate the current five-prompt execution plan and remain
historical only. `docs/NEXT_ACTION.md` names the sole active prompt and gate. After each phase,
update `PROJECT_STATE.md`, `DECISION_LOG.md` or `PROTOCOL_DEVIATIONS.md` as applicable,
`RESULTS_REGISTRY.md`, and `NEXT_ACTION.md`; run `scripts/check_project_context.py`; generate a
privacy-safe handoff; then move the finished
canonical prompt from `prompts/next/` to `prompts/completed/` and promote exactly one planned prompt.
