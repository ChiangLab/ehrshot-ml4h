# Next action

Status date: 2026-08-22

1. Review and commit the Prompt 2 implementation locally; generated aggregate and all private
   artifacts remain ignored.
2. Push the reviewed commit, reconnect to MedNet VPN, fast-forward the DGX checkout, and verify or
   download the pinned Qwen3-Embedding-0.6B artifact outside the scientific job chain.
3. Submit pilot prepare, throughput, task-array embedding, and fit/report jobs with `afterok`
   dependencies. The array concurrency remains one for the single physical GB10 GPU.
4. Verify every job/array sentinel, private cache COMPLETE marker, aggregate report, Notebook 04,
   and privacy-safe handoff before copying aggregate outputs to the Mac.
5. Do not construct the final-panel cohort, run M2/M3/8B, or inspect held-out performance.
6. After the verified handoff, await the user-supplied Prompt 3 of 5. Save it as
   `prompts/05_REVIEW_PILOT_AND_AUTHORIZE_NEXT_PHASE.md`; no file with that name exists yet.
