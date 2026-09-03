# Next action

Status date: 2026-09-03

Prompt 3 remains complete and held-out test performance remains locked. Canonical Prompt 4 M3
reference compute is complete at source commit `1a97d8f`: preparation 470_0-470_3 and seed jobs
471_0-471_11 passed. Do not rerun M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

M2 attempts 502, 507, and 513 are superseded and must remain preserved. Job 507_1 mechanically
completed 2,533 non-null readmission summaries, but a governed validation-only inspection exposed
repetition-driven output exhaustion, omitted facts, and unsupported code-name expansion. Job 513
reused zero rows and could not finish its 4,088 requests inside 12 hours. Cancellation was requested
for 507 and 513 on 2026-09-03; verify that both have left `squeue` before pulling new source.

The pending fidelity-v2 hotfix uses a new prompt/cache identity, compact structured output, exact
unexpanded codes, explicit recurrence, API `finish_reason=stop`, token metadata, restart-stable
served-model identity, and plain-code-aware fidelity parsing. Validate, commit, and push it; pull it
into a clean DGX checkout; export the four project/model/data variables; then run:

```bash
bash scripts/slurm/submit_prompt4_m2_fidelity_v2.sh
source outputs/private/prompt4_reference_job_ids.env
squeue -j "${M2_ARRAY_JOB_ID},${M2_EMBED_ARRAY_JOB_ID},${M2_REVIEW_PREP_JOB_ID}" \
  -o "%.18i %.12P %.30j %.2t %.10M %.10l %R"
```

The helper must first pass its dense synthetic live summary gate. It then reruns only M2 and records
new job IDs while retaining M3 470/471. Do not use any 502/507/513 cache row, submit sensitivity, or
access test data. After the new M2 summary/embedding tasks and review-packet job all have
COMPLETE/exit-0 evidence, perform the governed 40-item manual review on DGX. A failed fidelity gate
makes M2 ineligible but leaves M1/M3 valid. Only a passed manual and automated gate permits
reference finalization and validation-only approach selection.
