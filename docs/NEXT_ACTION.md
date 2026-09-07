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

After pulling fidelity-v7 source into a clean DGX checkout and exporting the project/model/data
variables, arm the unattended capacity-to-M2 chain:

```bash
bash scripts/slurm/arm_prompt4_m2_overnight_v7.sh
bash scripts/slurm/prompt4_overnight_status.sh
```

The capacity job chooses an error-free tier, and an `afterany` launcher fails closed unless that job
has a same-source COMPLETE sentinel. On success it submits the capacity-selected resumable M2 graph
and, independently, aggregates the already-complete M3 reference predictions without rerunning M3
training or attempting M1/M2/M3 selection early.

Do not combine legacy M2 prompts or exact-copy caches with fidelity-v6 summaries, and do not access
test data. M2 reference computation is complete only when all 768 summary shards, all four
embedding tasks, and the review-packet job have COMPLETE/exit-0 evidence. Then perform the private
40-item fidelity review and run `submit_prompt4_execution.sh reference-finalize` for validation-only
M2 fitting and the prespecified M1/M2/M3 approach selection.
