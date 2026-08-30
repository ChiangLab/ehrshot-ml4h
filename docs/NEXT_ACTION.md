# Next action

Status date: 2026-08-29

For the active `prompts/next/03_BUILD_TRUE_FOUR_TASK_FOUNDATION_AND_M1.md`, commit and push the
locally validated containerized-preflight hotfix, pull it into the clean DGX checkout, and submit
`scripts/slurm/submit_prompt3_execution.sh`. The complete 8B snapshot is present; the helper must
first pass its containerized hard gate and dependency-gated synthetic 8B smoke. Prompt 3 stays
active until all real jobs, atomic sentinels, semantic validators, hashes, and the privacy-safe
aggregate handoff pass. Do not run M2, M3, or held-out test evaluation.
