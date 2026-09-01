# Next action

Status date: 2026-09-01

Prompt 3 is closed by verified fit-recovery job 451. Preserve jobs 437, 444, 445_0 through 445_3,
failed job 446, and recovery job 451 unchanged. The active prompt is
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

Pull the committed Prompt 4 source into a clean DGX checkout, verify
`http://10.45.177.17:8000/v1/models`, export `TEMPORAL_EHR_PROJECT_ROOT` and
`TEMPORAL_EHR_MODEL_ROOT`, plus `EHRSHOT_ASSET_ROOT` and `EHRSHOT_DATA_PATH`, then run exactly:

```bash
bash scripts/slurm/submit_prompt4_execution.sh reference
```

Require COMPLETE/exit-0 sentinels for the M2 array/fit and M3 task-seed array/fit; validate
`m2_reference.json`, `m3_reference.json`, both executed notebooks, and
`approach_selection.json`. Only then run `bash scripts/slurm/submit_prompt4_execution.sh
sensitivity`; it submits only the selected approach's fixed temporal sensitivities (or no new model
when M1 is selected) and job 28's hashed freeze. Follow `docs/DGX_BATCH_RUNBOOK.md`. Do not access
held-out test data or performance.
