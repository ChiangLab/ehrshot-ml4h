# Next action

Status date: 2026-09-07

Prompt 4 is closed under the explicit deadline-preserving scope amendment. M1 direct serialization
is the sole primary approach for held-out evaluation. M3 reference validation is complete and is
retained only as secondary validation evidence. M2 is incomplete future work and cannot block,
enter, or alter the held-out analysis. No held-out test result has been accessed.

The sole active prompt is `prompts/next/05_ONE_TIME_TEST_AND_PAPER_ARTIFACTS.md`. The Prompt 5
implementation pins the completed M1 validation aggregate and M3 validation-only aggregate by
SHA-256, freezes validation-selected task penalties, records the train-only fitting rule, and uses
isolated submission-specific test artifacts. A private `STARTED.json` permanently records first
test access; a second experiment identity or a rerun after final completion is rejected.

After committing and pushing the reviewed source, pull that exact clean commit on the DGX, export
the usual project/model/data roots, and submit exactly once:

```bash
bash scripts/slurm/submit_m1_one_time_test.sh \
  --confirm-test-evaluation=I_CONFIRM_ONE_TIME_HELD_OUT_EVALUATION
```

The dependency chain freezes the manifest, constructs the official test-only cohort, prepares four
strictly censored task datasets, embeds each task with Qwen3-Embedding-0.6B, fits the frozen M1
heads and computes the fixed 2,000-resample patient bootstrap, then generates privacy-safe tables,
figures, manuscript text, claims, limitations, and a reproducibility manifest. Monitor without
printing patient data or performance:

```bash
bash scripts/slurm/prompt5_status.sh
```

Success requires COMPLETE/exit-0 sentinels for every stage, semantic validation, matching Git and
manifest hashes, the private prediction artifact hash, an aggregate final report with no row-level
fields, and `PUBLICATION_COMPLETE.json`. Never infer success from queue disappearance or file
existence alone. Do not resubmit the top-level script after the manifest is frozen. If an
intermediate stage fails, preserve it and resume the exact same manifest and verified caches with:

```bash
bash scripts/slurm/resume_m1_one_time_test.sh \
  --confirm-resume=I_CONFIRM_RESUME_SAME_FROZEN_HELD_OUT_EVALUATION
```

The recovery launcher refuses a changed commit, dirty worktree, missing original submission record,
or invalid manifest. It never freezes again and never reruns the held-out fit after a valid final
result exists.
