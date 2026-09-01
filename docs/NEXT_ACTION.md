# Next action

Status date: 2026-09-01

For the active `prompts/next/03_BUILD_TRUE_FOUR_TASK_FOUNDATION_AND_M1.md`, commit and push the
locally validated fit-provenance repair, then pull it into the clean DGX checkout. Preserve
foundation 437, throughput 444, embedding tasks 445_0 through 445_3, and failed fit 446 unchanged.
Submit only `scripts/slurm/23_final_m1_fit_validation.sbatch` through the fail-closed helper
`scripts/slurm/submit_prompt3_fit_recovery.sh`. Its preflight must validate all upstream source,
configuration, report, sentinel, and SHA-256 evidence and reject any partial fit output. Require the
new fit COMPLETE sentinel, final aggregate/notebook validators, and a privacy-safe handoff before
closing Prompt 3. Do not rerun any model, run M2/M3, or access held-out test evaluation.
