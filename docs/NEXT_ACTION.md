# Next action

Status date: 2026-09-06

Prompt 3 remains complete and held-out test performance remains locked. Canonical Prompt 4 M3
reference compute is complete at source commit `1a97d8f`: preparation 470_0-470_3 and seed jobs
471_0-471_11 passed. Do not rerun M3. The sole active prompt remains
`prompts/next/04_IMPLEMENT_M2_AND_M3.md`.

M2 attempts 502, 507, 513, 519, 529, 536, and 545 plus their canceled downstream jobs are
superseded and must remain preserved. Fidelity-v5 job 545 showed that its exact-copy requirement
was incompatible with genuine compression: six of the first eight shards rejected summaries that
exceeded the frozen 8,192-token embedding context, while two rejected non-exact model copies.
Shards 545_8/545_9 and downstream jobs 548/549 were canceled after the graph became unsatisfiable.
No M2 reference aggregate or model selection was authorized.

The active fidelity-v6 recovery restores the original Prompt 4 bounded single-pass-summary design.
It preserves the same grouped M1 source evidence, on-premise model, deterministic decoding,
8,192-token downstream context, fidelity thresholds, conditions, and test lock. A new 6,144-token
summary identity admits only `finish_reason=stop` responses that pass row-level code/value/unit/
recurrence/time-label checks and the downstream tokenizer. Three fixed same-configuration attempts
may recover nondeterministic contract failures without using outcomes. Sixteen shards per task,
four active shards, and two requests per shard keep eight total endpoint requests while reducing
each Slurm task's exposure to the 12-hour site limit. Summary-only shards reserve four CPUs and
16 GB each because inference remains on the separate vLLM machine.

Local validation passed (162 synthetic tests plus tracked-source Ruff, shell syntax, context, and
privacy/safety checks). Commit and push this source, pull it into a clean DGX checkout, export the
project/model/data variables, and run:

```bash
export PROMPT4_M2_SUMMARY_PARALLELISM=4
export PROMPT4_M2_EMBED_PARALLELISM=1
export PROMPT4_SLURM_MAIL_USER=wdelery@mednet.ucla.edu

bash scripts/slurm/submit_prompt4_m2_fidelity_v6.sh
source outputs/private/prompt4_reference_job_ids.env
bash scripts/slurm/prompt4_m2_progress.sh
```

The helper validates job-545 failure evidence, absence of downstream COMPLETE evidence, and
canonical M3 jobs 470/471. It then runs eight concurrent synthetic bounded-summary checks and
submits 64 resumable summary shards followed by dependent four-task embedding and manual-review
packet jobs. Do not submit sensitivity or access test data. After every summary/embedding task and
the review-packet job have COMPLETE/exit-0 evidence, perform the governed 40-item manual review on
DGX. Only a passed manual and automated fidelity gate permits reference finalization and
validation-only approach selection.
