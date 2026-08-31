# Next action

Status date: 2026-08-30

For the active `prompts/next/03_BUILD_TRUE_FOUR_TASK_FOUNDATION_AND_M1.md`, put the locally validated
throughput-reserve recovery on `origin/main` and pull it into the clean DGX checkout. Preserve the
COMPLETE foundation job 437 and FAILED throughput job 438. Submit only throughput, the embedding
array, and fit/report through `scripts/slurm/submit_prompt3_after_foundation.sh`; its preflight must
validate the exact job-437 sentinel, prior configuration hash, private foundation hashes, semantic
report, and the sole execution-only reserve change from 900 to 300 seconds. The required 2x runtime
contingency remains unchanged. Require all new atomic sentinels, validators, hashes, and the
privacy-safe aggregate handoff. Do not run M2, M3, or held-out test evaluation.
