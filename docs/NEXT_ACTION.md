# Next action

Status date: 2026-08-21

1. Off VPN, review and commit the current recovery/context change set with the suggested commit
   message `Recover job 407 and add durable project context`, then push `main` to GitHub.
2. Reconnect to MedNet VPN, SSH to `wdelery@10.45.62.90`, fast-forward the DGX checkout, and submit
   `scripts/slurm/11b_recover_phase1_alignment.sbatch` on the `batch` partition with
   `SOURCE_ALIGNMENT_JOB_ID=407` exactly as shown in `DGX_BATCH_RUNBOOK.md`.
3. Do not rerun Notebook 03. Verify scheduler COMPLETED/`0:0`, the log, the new recovery COMPLETE
   sentinel, the recovery validation record, artifact hashes, and the unchanged job-407 FAILED
   sentinel.
4. Build `reports/generated/handoff/project_handoff.json`, copy that and
   `reports/generated/phase1_protocol_review/protocol_alignment.json` back to the Mac, then
   disconnect from VPN.
5. Provide Prompt 2 of 5 together with the privacy-safe handoff bundle. Do not provide the executed
   notebook or any governed/private artifact.
