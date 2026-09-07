# Next action

Status date: 2026-09-07

Prompt 3 remains complete, canonical Prompt 4 M3 jobs 470/471 remain complete, and held-out test
performance remains locked. Do not rerun M1 or M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

Fidelity-v6 array 559 at commit `38e2b7b8` established valid row-level caching but inadequate
throughput. After approximately 11.6 hours, its four active anemia shards had accepted only 25-50
new rows each at approximately 3.4-5.5 rows/hour/shard and projected another 37-69 hours. The
remaining 60 shards and dependent embedding/review jobs 560/561 cannot establish timely reference
completion. Preserve every job-559 cache row, log, and sentinel; stop 559-561; do not resubmit the
same 16-shard-per-task graph.

The active fidelity-v7 recovery changes execution only. The M2 source evidence, Qwen model, prompt,
decoding, cache identity, fidelity thresholds, downstream 8,192-token embedding gate, task panel,
splits, and test lock remain unchanged. It uses a synthetic-only concurrency ladder at 8, 16, 24,
and 32 requests, chooses the highest error-free setting within 90% of maximum measured token
throughput, uses eight requests per shard, and divides each task into 192 deterministic shards.
Previously accepted fidelity-v6 rows are eligible only after the unchanged full row validator
passes again. The finer shards bound timeout loss; capacity-selected concurrency is the only source
of potential wall-time acceleration.

After committing and pushing fidelity-v7 source, pull it into a clean DGX checkout, export the
project/model/data variables, and submit the synthetic capacity job:

```bash
bash scripts/slurm/submit_prompt4_m2_capacity_v7.sh
source outputs/private/prompt4_m2_capacity_v7_job_id.env
squeue -j "${CAPACITY_JOB_ID}" -o "%.18i %.2t %.10M %.10l %R"
tail -f "slurm-tehr-p4-m2-cap-v7-${CAPACITY_JOB_ID}.out"
```

Require its COMPLETE sentinel and PASS report. Then submit the capacity-selected resumable graph:

```bash
bash scripts/slurm/submit_prompt4_m2_fidelity_v7.sh
source outputs/private/prompt4_reference_job_ids.env
bash scripts/slurm/prompt4_m2_progress.sh
```

Do not combine legacy M2 prompts or exact-copy caches with fidelity-v6 summaries, and do not access
test data. M2 reference computation is complete only when all 768 summary shards, all four
embedding tasks, and the review-packet job have COMPLETE/exit-0 evidence. Then perform the private
40-item fidelity review and run `submit_prompt4_execution.sh reference-finalize` for validation-only
M2 fitting and the prespecified M1/M2/M3 approach selection.
