# Next action

Status date: 2026-09-07

Prompt 4 is closed under the explicit deadline-preserving scope amendment. M1 direct serialization
is the sole primary approach for held-out evaluation. M3 reference validation is complete and is
retained only as secondary validation evidence. M2 is incomplete future work and cannot block,
enter, or alter the held-out analysis. The immutable manifest and deterministic test cohort are
now frozen and COMPLETE at jobs 572/573. No held-out embedding, prediction, or performance has
been produced.

The sole active prompt is `prompts/next/05_ONE_TIME_TEST_AND_PAPER_ARTIFACTS.md`. The Prompt 5
implementation pins the completed M1 validation aggregate and M3 validation-only aggregate by
SHA-256, freezes validation-selected task penalties, records the train-only fitting rule, and uses
isolated submission-specific test artifacts. A private `STARTED.json` permanently records first
test access; a second experiment identity or a rerun after final completion is rejected.

The initial graph must not be resubmitted. Preparation jobs `574_0`-`574_3` failed before
serialization because the shared preparation function retained its default train/validation guard.
Downstream jobs 575-577 were canceled. After committing and pushing the reviewed narrow repair,
pull the clean recovery commit on the DGX, export the usual project/model/data roots, and resume the
same frozen manifest exactly once:

```bash
bash scripts/slurm/resume_m1_one_time_test.sh \
  --confirm-resume=I_CONFIRM_RESUME_SAME_FROZEN_HELD_OUT_EVALUATION
```

The recovery validates the exact frozen source, job-574 failure signatures and hashes, existing
cohort/access evidence, and absence of final results. It then prepares four strictly censored task
datasets, embeds each task with Qwen3-Embedding-0.6B, fits the frozen M1 heads and fixed
2,000-resample patient bootstrap, and generates privacy-safe publication artifacts. Monitor without
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

The recovery launcher permits a different execution commit only through
`configs/recovery/prompt5_prepare_after_574.yaml`; it retains `7315451` as the frozen scientific
source and records the repair commit separately. It never freezes again, overwrites a 574 failure,
or reruns the held-out fit after a valid final result exists.
