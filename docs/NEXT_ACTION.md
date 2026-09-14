# Next action

Status date: 2026-09-14

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

1. On VPN, verify the DGX and on-prem Qwen3.6 endpoint, install the current clean source commit on
   the DGX, and run `scripts/slurm/arm_prompt4_m2_overnight_v7.sh`.
2. The wrapper must run a fresh synthetic 8/16/24/32 endpoint-capacity ladder before it submits the
   768 resumable M2 summary shards. Reuse only fidelity-v6/v7 rows that pass current row-level
   validation; generate only missing or invalid rows.
3. After all summary shards complete, allow the four task embedding jobs to use the unchanged
   frozen Qwen3-Embedding-0.6B encoder. Do not substitute the 8B encoder.
4. Complete the governed 40-history fidelity review before fitting or interpreting an M2
   validation classifier. Stop if any summary fidelity or downstream-token gate fails.
5. Do not run M3 computation again and do not access or rerun held-out evaluation. Any M2 result is
   validation-only and must be labeled post-completion exploratory evidence relative to the frozen
   M1 study.

The prior launcher job 568 failed before M2 submission because an early-exiting `awk` pipeline ran
under `pipefail` and propagated SIGPIPE exit 141. The submission script now consumes the complete
Slurm configuration stream before reading `MaxArraySize`; a focused regression test protects this
boundary. The restart wrapper also keeps the current checkout commit separate from any historical
armed-job record, so stale job-568 metadata cannot authorize reuse of old capacity evidence.

The authoritative aggregate artifacts are under
`reports/generated/submission/locked_four_task_m1_submission_v1/`; the privacy-safe handoff is
`reports/generated/handoff/prompt5_submission_handoff.json`.
