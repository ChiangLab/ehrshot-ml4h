# Next action

Status date: 2026-09-05

Prompt 3 remains complete and held-out test performance remains locked. Canonical Prompt 4 M3
reference compute is complete at source commit `1a97d8f`: preparation 470_0-470_3 and seed jobs
471_0-471_11 passed. Do not rerun M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

M2 attempts 502, 507, 513, 519, 529, and 536 plus their canceled downstream jobs are superseded and
must remain preserved. At commit `8682cb4`, shards 536_0/536_1 saved 400/375 new exact summaries
but rejected 49/34 length-limited responses at the 8,192-token instruction-generation ceiling.
No M2 reference aggregate or model selection was authorized.

The active fidelity-v5 recovery keeps the prompt, grouping transform, and both prior 4,096/8,192
identities unchanged. It reconstructs each source and reuses a candidate only when the assistant
text is an exact copy and passes the frozen downstream tokenizer. Missing/rejected rows use a new
16,384-token instruction-generation hash while the downstream embedding context remains 8,192.
Eight deterministic shards per task run with at most two active and report rate/ETA every 25 rows.

Cancel job 536 and wait for it to leave the queue. Validate, commit, and push the fidelity-v5
source; pull it into a clean DGX checkout; export the
project/model/data variables; then run:

```bash
bash scripts/slurm/submit_prompt4_m2_fidelity_v5.sh
source outputs/private/prompt4_reference_job_ids.env
squeue -j "${M2_ARRAY_JOB_ID},${M2_EMBED_ARRAY_JOB_ID},${M2_REVIEW_PREP_JOB_ID}" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
```

The helper validates the required job-536 length failures, all observed final 536 sentinels, the
absence of downstream COMPLETE evidence, and canonical M3 470/471. It then passes three exact
synthetic 16,384-budget generations through the unchanged downstream tokenizer gate and submits 32
bounded summary shards followed by dependent embedding/review jobs. Do not submit sensitivity or
access test data. After the new M2 summary/embedding tasks and review-packet job all have
COMPLETE/exit-0 evidence, perform the governed 40-item manual review on DGX. A failed fidelity gate
makes M2 ineligible but leaves M1/M3 valid. Only a passed manual and automated gate permits
reference finalization and validation-only approach selection.
