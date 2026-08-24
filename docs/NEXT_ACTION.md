# Next action

Status date: 2026-08-24

1. Fast-forward the clean DGX checkout to commit `cb07cb5` or its workflow-only descendant, restore
   the four governed path variables, and run `bash scripts/slurm/submit_prompt2_execution.sh` once.
   Do not redownload the complete model and do not treat failed smoke job 411 as success.
2. After all dependency-locked jobs stop, verify the new smoke report, every job/array sentinel,
   every private cache COMPLETE marker, the aggregate pilot report, Notebook 04, and the
   privacy-safe handoff before copying only aggregate outputs to the Mac.
3. Do not construct the final-panel cohort, run M2/M3/8B, or inspect held-out performance until the
   Prompt 2 pilot is COMPLETE.
4. Once its aggregate and handoff are copied locally, execute the sole active canonical prompt:
   `prompts/next/03_BUILD_TRUE_FOUR_TASK_FOUNDATION_AND_M1.md`.
