# Next action

Status date: 2026-09-04

Prompt 3 remains complete and held-out test performance remains locked. Canonical Prompt 4 M3
reference compute is complete at source commit `1a97d8f`: preparation 470_0-470_3 and seed jobs
471_0-471_11 passed. Do not rerun M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

M2 attempts 502, 507, 513, and 519 plus canceled downstream jobs 520/521 are superseded and must
remain preserved. At commit `a316c98`, 519_0 and 519_1 reached 12 hours, 519_2 was canceled, and
519_3 rejected 549/650 responses with `finish_reason=length`. No M2 reference aggregate or model
selection was authorized.

The active fidelity-v3 recovery deterministically groups facts within each unchanged window by
code/source/unit, preserves distinct numeric/text values in first-occurrence order and exact total
occurrences, and makes Qwen copy that compact structure. It uses a new prompt, transform, and cache
identity; raises only the output ceiling from 2,048 to 4,096; requires `finish_reason=stop`; checks
every summary with the offline 8,192-token embedding tokenizer; and reports progress every 25 rows.

Validate, commit, and push the fidelity-v3 source; pull it into a clean DGX checkout; export the
project/model/data variables; then run:

```bash
bash scripts/slurm/submit_prompt4_m2_fidelity_v3.sh
source outputs/private/prompt4_reference_job_ids.env
squeue -j "${M2_ARRAY_JOB_ID},${M2_EMBED_ARRAY_JOB_ID},${M2_REVIEW_PREP_JOB_ID}" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
```

The helper must first validate all job-519 failures, the absence of downstream COMPLETE evidence,
and canonical M3 470/471, then pass three exact dense synthetic live summaries plus the downstream
tokenizer gate. It reruns only M2 and records new job IDs. Do not use any prior M2 row under the new
cache identity, submit sensitivity, or access test data. After the new M2 summary/embedding tasks and review-packet job all have
COMPLETE/exit-0 evidence, perform the governed 40-item manual review on DGX. A failed fidelity gate
makes M2 ineligible but leaves M1/M3 valid. Only a passed manual and automated gate permits
reference finalization and validation-only approach selection.
