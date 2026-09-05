# Next action

Status date: 2026-09-05

Prompt 3 remains complete and held-out test performance remains locked. Canonical Prompt 4 M3
reference compute is complete at source commit `1a97d8f`: preparation 470_0-470_3 and seed jobs
471_0-471_11 passed. Do not rerun M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

M2 attempts 502, 507, 513, 519, and 529 plus their canceled downstream jobs are superseded and must
remain preserved. At commit `818103e`, all four 529 tasks ended FAILED: 529_0/529_1 reached the
12-hour limit, 529_2 was intentionally canceled after caching additional rows, and 529_3 rejected
333 CheXpert responses at the 4,096-token completion ceiling. No M2 reference aggregate or model
selection was authorized.

The active fidelity-v4 recovery keeps the fidelity-v3 prompt, grouping transform, and primary
4,096-token identity unchanged. It reconstructs each source and reuses a candidate only when the
assistant text is an exact copy and passes the frozen downstream tokenizer. Missing/rejected rows
use a separate 8,192-token decoding hash. Eight deterministic shards per task run with at most two
active and report rate/ETA every 25 rows; each successful response is cached atomically.

Validate, commit, and push the fidelity-v4 source; pull it into a clean DGX checkout; export the
project/model/data variables; then run:

```bash
bash scripts/slurm/submit_prompt4_m2_fidelity_v4.sh
source outputs/private/prompt4_reference_job_ids.env
squeue -j "${M2_ARRAY_JOB_ID},${M2_EMBED_ARRAY_JOB_ID},${M2_REVIEW_PREP_JOB_ID}" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
```

The helper first validates all job-529 failures, the absence of downstream COMPLETE evidence, and
canonical M3 470/471, then passes three exact synthetic 8,192-budget summaries plus the downstream
tokenizer gate. It submits 32 bounded summary shards, then dependent four-task embedding and review
jobs. Do not submit sensitivity or access test data. After the new M2 summary/embedding tasks and review-packet job all have
COMPLETE/exit-0 evidence, perform the governed 40-item manual review on DGX. A failed fidelity gate
makes M2 ineligible but leaves M1/M3 valid. Only a passed manual and automated gate permits
reference finalization and validation-only approach selection.
