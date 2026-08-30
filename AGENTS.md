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
3 source now implements the new versioned final cohort, Qwen3-Embedding-0.6B at 8,192 tokens,
final M1 and B1-B4, the runtime-only 8B feasibility gate, and train/validation only. The source is
committed at `6a0d0accddb0098f713796ca65750ca162a7daeb`, and no final real-data/model job has run. The
complete hash-manifested Qwen3-Embedding-8B snapshot was transferred to the DGX. A subsequent
host-side preflight exposed missing host NumPy before any jobs were submitted; commit `fe30dda`
moved this package-dependent gate inside the NVIDIA container. After that fix, synthetic 8B
smoke job 423 passed on `dgit00038673`, but foundation job 424 failed before scientific execution
on `dgit00038674` because Pyxis could not create `/run/user/2051`; downstream jobs 425–427 were
canceled. Pin Prompt 3 container work to the verified node until the second node is repaired. M2,
M3, and held-out evaluation remain prohibited.

The completed Prompt 2 authorization supersedes the sequencing and narrower scope proposed in
`prompts/03_BUILD_PHASE3_BASELINES.md` and `prompts/04_BUILD_THROUGHPUT_AND_M1_PILOT.md`. Keep those
files as historical planning records. The sole active implementation prompt is
`prompts/next/03_BUILD_TRUE_FOUR_TASK_FOUNDATION_AND_M1.md`.

Canonical phase prompts live under `prompts/completed/`, `prompts/next/`, and `prompts/planned/`.
Files numbered 03 and 04 at the top level predate the current five-prompt execution plan and remain
historical only. `docs/NEXT_ACTION.md` names the sole active prompt and gate. After each phase,
update `PROJECT_STATE.md`, `DECISION_LOG.md` or `PROTOCOL_DEVIATIONS.md` as applicable,
`RESULTS_REGISTRY.md`, and `NEXT_ACTION.md`; run `scripts/check_project_context.py`; generate a
privacy-safe handoff; then move the finished
canonical prompt from `prompts/next/` to `prompts/completed/` and promote exactly one planned prompt.
