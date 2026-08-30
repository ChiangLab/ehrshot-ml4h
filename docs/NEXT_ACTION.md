# Next action

Status date: 2026-08-29

Review the uncommitted implementation of
`prompts/next/03_BUILD_TRUE_FOUR_TASK_FOUNDATION_AND_M1.md`, then create and push one clean source
commit. On the DGX, install and smoke-test the complete pinned Qwen3-Embedding-8B snapshot before
submitting `scripts/slurm/submit_prompt3_execution.sh`. Prompt 3 stays active until all real jobs,
atomic sentinels, semantic validators, hashes, and the privacy-safe aggregate handoff pass. Do not
run M2, M3, or held-out test evaluation.
