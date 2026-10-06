# Next action

## Current override: deploy bounded-omission v4, then review

Follow `docs/M2_SELECTION_V4_RECOVERY.md`. Publish the scoped recovery commit, pull it on an idle
DGX, then run `bash scripts/slurm/submit_m2_selection_v4.sh start`. The helper sets paths and
queues only after the frozen source, old failure and whole-source token feasibility checks pass.
Canary -> independent confirmation -> readiness -> bounded shards -> seal/review packet ->
embeddings are automatically gated. Review all 40 histories before explicit `finalize` and final
semantic verification. Never remove HALT or rerun a failed scientific gate blindly.
The reference run does not complete the full M2 condition/sensitivity matrix. Plan that only after
reference review, without reopening held-out testing. All older M2 launch instructions below
are superseded. No active implementation prompt remains.

Status date: 2026-09-30

## Current next action: selection-v3 recovery after array 993

Do not resume `m2_structured_v2_reference_v1` or delete its HALT. Job 993 stopped correctly after
two generated summaries failed structural fidelity; an identical rejected response at temperature
zero makes another unchanged retry unsound. Follow `docs/M2_SELECTION_V3_RECOVERY.md`.

On the Mac, publish only the named recovery with
`bash scripts/git/publish_m2_selection_v3.sh`. On a clean, idle DGX, pull that commit and run
`bash scripts/slurm/submit_m2_selection_recovery.sh start`. The graph is canary -> independent
confirmation -> read-only readiness -> resumable production -> coverage/review packet -> four
0.6B embedding jobs. Monitor with `scripts/slurm/m2_selection_v3_status.sh`. The model returns
only ordered source-series indices; code materializes exact clinical facts and reapplies all
retention/time/token gates. Old caches cannot satisfy the new keys.

After coverage, personally complete the governed 40-history review with
`scripts/review_m2_selection_v3_packet.py`. Only after review and embeddings pass, run the
launcher with `finalize`, then `scripts/slurm/verify_m2_selection_v3.sh`. A new HALT is a hard
stop, not a file to remove. M1, M3 and the one-time held-out test remain closed.

## Superseded next action: structured-v2 reference production

Jobs 988/989/990 passed; do NOT resubmit engineering. Follow
`docs/M2_STRUCTURED_V2_PRODUCTION.md`. On the Mac, run
`bash scripts/git/publish_m2_structured_production.sh`; on a clean idle DGX after pulling that
commit, run `bash scripts/slurm/submit_m2_structured_production.sh start`. It sets required paths,
revalidates engineering, freezes source/worklists, and queues generation -> coverage/review packet
-> embeddings. Monitor `scripts/slurm/m2_production_status.sh` from the repository.

After coverage, review the 40 histories privately with `python3 scripts/review_m2_structured_packet.py`.
After all embeddings and review, run the same launcher with `finalize`, then
`bash scripts/slurm/verify_m2_production.sh`. Copy reviewed aggregate reports/handoff only.
An interrupted infrastructure job may use `resume` after the queue is empty; a fidelity HALT
requires investigation, never deletion or bypass. No scientific production success is claimed yet.
Further M2 sensitivity and manuscript integration follow reference review; M1/M3/test stay closed.

For governed presentation review, deploy the optional localhost-only browser interface after the
same commit is on the DGX: `docs/PROJECT_EXPLORER.md`. It builds a private train/validation DuckDB
index and never authorizes test access. Rebuild it after M2 advances to refresh M2 views.

## Active override: versioned generation repair after 986

Use `docs/M2_STRUCTURED_V2_AFTER_986.md`, not the older launchers below. Publish the scoped source
from the Mac with `scripts/git/publish_m2_structured_v2.sh`, pull into a clean idle DGX checkout,
then submit once with `scripts/slurm/submit_m2_structured_v2.sh`. This is a 64-input regression,
32-input fresh confirmation and readiness chain, not full-cohort M2. Monitor with
`scripts/slurm/m2_structured_v2_status.sh`; verify all stages recursively with
`scripts/slurm/verify_m2_structured_v2.sh`. If any gate fails, preserve evidence and stop.
Return only the aggregate reports under `reports/generated/engineering/m2_structured_v2/`.
The next production worker/cache integration requires the verified new engineering evidence;
embeddings, frozen manual review and validation fitting are not yet run. M1/test stays closed.

## Active override: recover 982's rendering, then confirm

Follow `docs/M2_RENDER_V2_RECOVERY.md`. All 32 latest saved development responses passed factual
checks; the two oversized rendered outputs were 56/73 tokens over 8,192. The compact renderer
must round-trip exactly and pass the same real tokenizer cap. Preserve every original artifact.

1. Validate and deploy only the scoped rendering-recovery source. No changes are committed/pushed
   by the implementation assistant.
2. On a clean, idle DGX, run `scripts/slurm/submit_m2_render_recovery.sh` once. It submits a
   30-minute replay of saved 982 responses with no generation, then a dependent four-hour job
   for the independent confirmation sample frozen before job 982.
3. Monitor `scripts/slurm/m2_render_status.sh`; require COMPLETE plus semantic verification for
   both jobs. The second job may have no sentinel if the first fails before it starts.
4. Return only the aggregate reports under `reports/generated/engineering/m2_render_v2/`.
   Do not repeat the old development job, skip oversized rows, or enlarge the encoder context.
5. Review those gates before production integration/capacity sizing. Full train/validation M2
   generation, 0.6B embeddings, frozen 40-history manual review, and validation fitting remain.
   The one-time M1 test remains closed; M3 470/471/570 is preserved.

## Prior action: ordered structured M2 (superseded above)

Job 980 at `0d6ad24` failed (24/32 final PASS). William authorized a separate ordered structured
summary version, not another unchanged legacy retry. Follow `docs/M2_STRUCTURED_V1_RUNBOOK.md`.

1. Review the scoped source changes and local synthetic/Ruff/shell/privacy/context checks.
   This implementation turn leaves changes uncommitted and submits no DGX work.
2. After deployment to a clean DGX checkout, use `submit_m2_structured_gates.sh`. It submits a
   32-input training development job, then a 32-input patient-disjoint confirmation job dependent
   on development success. No full-generation, embedding, fit, M3, or test job is submitted.
3. Monitor with `m2_structured_status.sh`. Require both COMPLETE sentinels, exit 0, report PASS,
   manifest hashes and independent semantic revalidation. A failure blocks the next gate; retain
   diagnostics privately. Never infer completion from an empty queue.
4. Return only the two aggregate JSON reports. Inspect fidelity, compression, token budgets and
   measured runtime before implementing/authorizing the full version-specific production graph.
   Old accepted summaries cannot satisfy new-version keys. No performance-based design changes.
5. Production still requires complete train/validation coverage, row-level checks, downstream
   0.6B embeddings, frozen 40-history manual fidelity review, and validation-only fitting. The new
   path has not completed these. Do not use the manual-review sample to tune the prompt.

This explicitly changes source representation/prompt/schema, not clinical tasks, retained clinical
coverage, instruction model, 0.6B encoder, or M1/M3/test results. No fact-ID/source-rendered fallback
is substituted. Ordered measurement series are preserved; global cross-code interleaving is not.

## Historical finalization and singleton-recovery instructions (superseded above)

All five implementation prompts are complete. No active implementation prompt remains. The
one-time M1 held-out evaluation and publication bundle completed under frozen scientific source
commit `731545119df7a313883e60967039b5e0ddd2a2f1` and recovery execution commit
`e9f8f5901037ddc8f5a6a5ea42019275fae0fd9e`.

Jobs 572 and 573 froze the manifest and constructed the deterministic held-out cohort. The original
preparation jobs `574_0`-`574_3` remain preserved failures. Recovery jobs `583_0`-`583_3` and
`585_0`-`585_3`, followed by jobs 586 and 587, all wrote COMPLETE and passed semantic validation.
The held-out aggregate SHA-256 is
`9c8af6cab8d1ec48e168b6ea6c1f045bf6dc4705f6f7ceccc21a8b7fabb8031f`; the publication sentinel
SHA-256 is `e0e06c759379e9b9bd08fc78ce21114aef853eb71fe366a8065d3129dfc38c3c`.

The completed M1 result remains frozen. Following Jeff's request to continue the previously deferred
summarization experiment, the next computational action is a validation-only M2 resumption. It is a
post-completion extension and must not access held-out outcomes or change any M1 analysis.

1. Do not re-arm fidelity-v7 or rerun 768 shards yet. The user reported array 596 stopped with
   12 COMPLETE, 369 FAILED, and 387 unfinished shards. Preserve 596/597/598 and all caches.
2. Deploy the scoped singleton-validator repair, then follow `docs/M2_JOB596_RECOVERY.md` to
   submit only `scripts/slurm/submit_m2_after596_canary.sh`. It evaluates 32 training histories
   spanning four tasks, two temporal conditions, and four source-length strata. It does not use
   the frozen 40-history validation review sample, embeddings, labels, or outcomes for selection.
3. Require the canary COMPLETE sentinel AND semantic/hash verification. A FAIL produces private
   diagnostic responses and a privacy-safe aggregate; do not bypass it. A PASS is engineering
   evidence only, not permission to skip manual fidelity review or a guarantee of full-run success.
4. Review that evidence before authorizing the next full M2 recovery graph and fresh capacity
   check. Reuse only exact, revalidated summary caches; retain Qwen3-Embedding-0.6B. The planned
   graph remains summaries -> embeddings and review packet -> manual review -> validation fit.
5. Do not rerun M3 or access/rerun held-out evaluation. M2 remains a post-completion exploratory
   validation-only extension; no new M1 selection or test freeze is authorized.

The source formatter supplies `occurrences=1`, but the old output grammar rejected that exact
field. The repair accepts explicit positive singleton counts while retaining source-matching,
recurrence retention, numeric/unit association, timing, trend, and token gates. The 333 recurrence
categories were first-error log classifications, not proof of 333 incorrect clinical counts.
The canary must establish which failures remain. No extractive/fact-ID redesign is implemented.

The prior launcher job 568 failed before M2 submission because an early-exiting `awk` pipeline ran
under `pipefail` and propagated SIGPIPE exit 141. The submission script now consumes the complete
Slurm configuration stream before reading `MaxArraySize`; a focused regression test protects this
boundary. The restart wrapper also keeps the current checkout commit separate from any historical
armed-job record, so stale job-568 metadata cannot authorize reuse of old capacity evidence.

The authoritative aggregate artifacts are under
`reports/generated/submission/locked_four_task_m1_submission_v1/`; the privacy-safe handoff is
`reports/generated/handoff/prompt5_submission_handoff.json`.
