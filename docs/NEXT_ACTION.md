# Next action

Status date: 2026-09-01

Prompt 3 is closed by verified fit-recovery job 451. Preserve jobs 437, 444, 445_0 through 445_3,
failed job 446, and recovery job 451 unchanged. The active prompt is
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

First cancel and preserve the superseded commit-`a28eab1` Prompt 4 graph (456, 458, 459, 460);
456_0 already failed and no output from that graph may satisfy final gates. Do not delete its
summaries, embeddings, logs, or sentinels. The aligned source safely reuses only immutable summary
rows and writes source-commit-isolated embeddings.
Preserve failed preparation array 465 and dependency-blocked seed array 469. Commit/pull the
`CONTAINER_IMAGE` hotfix, cancel 469, and submit new M3 preparation/training IDs; none of 465/469
may satisfy Prompt 4 gates.

After local revalidation and review/commit of the Prompt 4 alignment update, pull it into a clean
DGX checkout, verify
`http://10.45.177.17:8000/v1/models`, export `TEMPORAL_EHR_PROJECT_ROOT` and
`TEMPORAL_EHR_MODEL_ROOT`, plus `EHRSHOT_ASSET_ROOT` and `EHRSHOT_DATA_PATH`, then run exactly:

```bash
bash scripts/slurm/submit_prompt4_capacity.sh
bash scripts/slurm/prompt4_status.sh
```

Use only synthetic capacity evidence to confirm or revise M2 concurrency and M3 embedding batch
size or operational array parallelism before scientific compute. Then run
`bash scripts/slurm/submit_prompt4_execution.sh reference`. M3 preparation must complete once for
each task before the 12 task/seed fits; require COMPLETE/exit-0 for the M2 summary/embedding arrays,
M3 preparation/training arrays, and private review-packet job. Complete the governed 40-history
manual review, then run
`bash scripts/slurm/submit_prompt4_execution.sh reference-finalize`. Validate `m2_reference.json`,
`m2_manual_fidelity_review.json`, `m3_reference.json`, both notebooks, and
`approach_selection.json`. Only then run `bash scripts/slurm/submit_prompt4_execution.sh
sensitivity`; it includes the selected approach's B1 and C3 co-primary cells before job 28 freezes
the hashed manifest. Follow `docs/DGX_BATCH_RUNBOOK.md`. Do not access held-out test performance.
