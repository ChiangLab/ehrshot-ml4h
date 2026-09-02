# Next action

Status date: 2026-09-02

Prompt 3 remains complete and held-out test performance remains locked. Canonical Prompt 4 M3
reference compute is complete at source commit `1a97d8f`: preparation 470_0-470_3 and seed jobs
471_0-471_11 passed. Do not rerun M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

M2 job 502_0 is invalid and preserved with its original RUNNING sentinel. Its old cache rows cannot
satisfy the new decoding hash. Validate, commit, and push the M2 recovery source, pull it into a
clean DGX checkout, export `TEMPORAL_EHR_PROJECT_ROOT`, `TEMPORAL_EHR_MODEL_ROOT`,
`EHRSHOT_ASSET_ROOT`, and `EHRSHOT_DATA_PATH`, then run:

```bash
bash scripts/slurm/submit_prompt4_m2_recovery.sh
source outputs/private/prompt4_reference_job_ids.env
bash scripts/slurm/prompt4_status.sh 502_0
```

The preflight must pass before it submits four new M2 summary tasks, four dependent embedding
tasks, and the manual-review packet. When they all have COMPLETE/exit-0 and no FAILED sentinel,
perform the governed 40-item manual fidelity review on DGX and run:

```bash
bash scripts/slurm/submit_prompt4_execution.sh reference-finalize
```

That stage fits M2, aggregates the already-computed M3 evidence with explicit cross-commit
provenance, and applies the frozen validation-only approach-selection rule. Only after its semantic
review passes may the selected-approach sensitivity/freeze stage run. Do not access test data.
