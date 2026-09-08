# Next action

Status date: 2026-09-08

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

The next action is manuscript and submission finalization, not additional modeling:

1. Report the observed held-out results without changing the protocol. C1 equal-weight
   macro-AUROC was 0.646. C1 did not outperform latest-window or shuffled-time labeling, and the
   code-count baseline had the highest macro-AUROC among the reported representations.
2. Keep M3 as validation-only secondary evidence and M2 as incomplete future work. Neither may be
   added to held-out evaluation after seeing the M1 result.
3. Resolve the author list, institutional ethics/IRB wording, code-availability language, and
   official ML4H template/page-limit checks.
4. Cross-check every number in the manuscript and slides against the frozen publication CSV/JSON
   files. Do not use private rows or rerun the held-out workflow.

The authoritative aggregate artifacts are under
`reports/generated/submission/locked_four_task_m1_submission_v1/`; the privacy-safe handoff is
`reports/generated/handoff/prompt5_submission_handoff.json`.
