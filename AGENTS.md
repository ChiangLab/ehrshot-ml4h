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
report without rerunning it. Job 408 used commit
`e12d475d5ec2bc1f6c02aeda28e1a252ac84d6ea`, wrote an atomic COMPLETE sentinel with exit status 0,
and preserved job 407's FAILED sentinel. Historical `sacct` is unavailable on this site, so durable
sentinels, the clean log, hashes, and semantic validation are the retained evidence.

The final four-task panel is prospectively LOCKED, but it is isolated from the authorized
three-task engineering PILOT. A pilot-mode gate may authorize only the pilot artifact namespace;
it must never allow a pilot cohort or cache to satisfy a final-panel gate. Current implementation
scope is Qwen3-Embedding-0.6B, 4,096 tokens, M1 and B1-B4, train/validation only. M2, M3, 8B, the
final-panel cohort, and test evaluation remain prohibited.

The current Prompt 2 authorization supersedes the sequencing and narrower scope proposed in
`prompts/03_BUILD_PHASE3_BASELINES.md` and `prompts/04_BUILD_THROUGHPUT_AND_M1_PILOT.md`. Keep those
files as historical planning records; do not execute them as the next action. After the verified
pilot handoff, wait for the user-supplied Prompt 3 of 5, to be saved as
`prompts/05_REVIEW_PILOT_AND_AUTHORIZE_NEXT_PHASE.md`.
