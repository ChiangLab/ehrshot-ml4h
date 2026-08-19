# Codex prompt: audit Phase 1 and align the implementation with the final protocol

Work in the current `temporal-ehr-ml4h` repository. This is a scientific-protocol alignment task,
not a model-training task. Read the repository before editing. In particular, read:

- `README.md`
- `configs/tasks.yaml`
- `configs/expected_ehrshot.yaml`
- `docs/AUG14_16_RUNBOOK.md`
- `docs/TASK_SELECTION_REVIEW.md`
- `prompts/01_BUILD_PHASE1_NOTEBOOKS.md`
- all modules under `src/temporal_ehr/serialization/`, `src/temporal_ehr/data/`, and
  `src/temporal_ehr/evaluation/`
- all tests
- `scripts/slurm/10_run_phase1_notebooks.sbatch`
- the final protocol, if this path is available:
  `../../temporal representation/FINAL_TEMPORAL_EHR_PROJECT_PROTOCOL_2026-08-14.md`
- the execution runbook, if this path is available:
  `../../temporal representation/EXECUTION_RUNBOOK_2026-08-14_TO_2026-09-10.md`

If either parent-directory document is unavailable, continue using the governing decisions below.
Do not infer missing scientific decisions.

## Goal

Bring the serializer, temporal perturbations, packing logic, job runner, and documentation into
alignment with the final temporal-EHR protocol before any embedding experiment is run. Preserve the
working Phase 1 foundation and count baseline. Make the implementation task-panel agnostic, record
the unresolved task-panel decision explicitly, and produce a privacy-safe review notebook.

Do **not** download a model, generate embeddings, fit an embedding-based model, inspect test
performance, commit, or push.

## Current state that must be preserved and accurately documented

- The event audit, label validation, governed cohort construction, strict censoring, and count
  baseline have run successfully on real EHRSHOT data.
- The current governed cohort and notebooks use a three-task **pilot** panel:
  `lab_anemia`, `new_acutemi`, and `guo_los`.
- Current validation-only count-reference results exist under
  `reports/generated/count_baseline/`; do not fabricate or hardcode them.
- The three source notebooks contain no output. Executed copies are private.
- The repository currently passes its tests and privacy checks.
- A DGX Slurm job executed notebooks 00, 01, and 02, but the old runner did not write a formal
  completion sentinel. Treat the final scheduler state as unverified until `sacct` and the log are
  checked.

## Unresolved task-panel decision: do not silently choose

The final written protocol proposed this four-task breadth panel:

1. `lab_anemia`
2. `guo_readmission`
3. `new_hypertension`
4. `chexpert`

The implemented pack instead uses the three-task pilot listed above. That difference is a protocol
decision, not a coding error to silently overwrite. Create:

- `configs/experiments/pilot_three_task.yaml`, clearly marked `status: PILOT`
- `configs/experiments/proposed_four_task.yaml`, clearly marked `status: PROPOSED`
- `configs/experiments/locked_panel.yaml`, initially marked `status: PENDING`

The locked file must explain that the user should copy exactly one reviewed panel into it and change
the status to `LOCKED` only after confirming the choice with Jeff. Downstream training code must
fail closed unless `status: LOCKED`. Do not rebuild or overwrite the existing governed cohort in
this task.

Create `docs/PROTOCOL_DEVIATIONS.md` with a table containing, at minimum:

- the unresolved three-task versus four-task panel difference;
- strict universal `event_time < prediction_time` censoring versus any task-specific boundary in
  the formal benchmark, including the rationale that exact-prediction-time events were observed;
- the current deterministic cohort-selection hash versus the precise final-protocol formulation;
- the fact that aggregate test split counts/prevalence were previously produced during data QC,
  while no test predictions or test performance were used; and
- every implementation difference discovered during this audit.

For each item record: identifier, protocol expectation, current behavior, risk, proposed resolution,
decision owner, status, and date. Do not call a deviation “resolved” without evidence.

## Required scientific corrections

### 1. Prediction-relative clinical blocks

The primary experiment must operate on chronological prediction-relative windows, not calendar-day
blocks. Keep calendar-day blocks only as an intermediate representation and QC view.

- Use exact elapsed time from each event timestamp to the prediction timestamp.
- Use half-open, documented interval boundaries so an event cannot enter two windows.
- Use 30-day windows for laboratory, imaging, and operational tasks.
- Use 90-day windows for incident-diagnosis tasks.
- Key the window size from the task family in configuration; never guess from a task name.
- Preserve stable deterministic ordering within a window without implying clinical order among
  truly concurrent events.
- Add boundary tests for events exactly at 0, 30, 60, 90, and 180 days, including time-of-day cases.

### 2. Exact duplicate handling

Remove exact duplicate events before rendering or recurrence analysis. Define the duplicate key
explicitly from the available EHRSHOT fields and justify it in documentation. Do not collapse
clinically recurrent events merely because their codes repeat. Report aggregate duplicate-removal
counts by task and split with small-cell suppression.

### 3. Explicit compact-time labels

Replace ambiguous labels such as `[TIME 37 DAYS BEFORE]` with stable range labels tied to the
configured window, for example:

- `[TIME=1_30_DAYS_BEFORE_T]`
- `[TIME=31_60_DAYS_BEFORE_T]`
- `[TIME=61_90_DAYS_BEFORE_T]`

Define `T` as the prediction time. The most recent retained block must always carry its correct
label. Do not include calendar dates or patient timestamps in model text.

### 4. Four complete-history temporal conditions

All four conditions must contain the same retained clinical events in the same event order. Only
the time interface may differ:

- `C0_order_only`: chronological blocks, no elapsed-time labels.
- `C1_correct_compact_time`: correct prediction-relative labels.
- `C2_reversed_earlier_gaps`: keep clinical blocks and events in chronological order; reverse the
  sequence of inter-block gaps among earlier blocks; reconstruct monotonic falsified labels; keep
  the final block's label correct.
- `C3_shuffled_earlier_time_labels`: keep clinical blocks and events in chronological order; permute
  earlier time labels using three prespecified fixed seeds; keep the final label correct.

For C2, directly test that the earlier gap multiset is preserved, labels remain monotonic, the total
history span is preserved when mathematically applicable, the final label is correct, and at least
one eligible earlier label changes. If a history cannot satisfy the perturbation, mark it ineligible
with a reason; never fabricate a change.

For C3, directly test all three fixed seeds, deterministic replay, label-multiset preservation,
correct final label, in-range labels, unchanged event order/content, and a changed eligible earlier
mapping. Store the seeds in configuration, not hidden notebook state.

### 5. Content-matched primary packing

The primary C0-C3 comparison must not allow time tokens to displace different clinical evidence.
Implement two clearly named policies:

1. `content_matched_primary`: determine the shared most-recent clinical suffix that fits the token
   budget in the most token-expensive eligible condition, then render exactly that same clinical
   content in C0-C3.
2. `independent_operational`: independently pack each condition to the budget and report how many
   clinical events time tokens displace. This is an engineering sensitivity analysis, not the
   primary causal contrast.

At 4,096 and 8,192 Qwen tokens, assert for the content-matched policy that patient/task/instance,
outcome, retained exact-event hashes, block order, and event order are identical across C0-C3.
Only allowed time-label fields may differ. Store private per-instance manifests under
`outputs/private/`; write only aggregate QC under `reports/generated/`.

### 6. Separate information-reduction baselines

Implement renderers and invariants for these baselines, but do not embed them yet:

- `latest_window_only`: retain the entire most recent 30- or 90-day observation window, depending
  on task family. This follows Jeff's clarification and is not a second row of a 2-by-4 matrix.
- `latest_code_occurrence`: retain only the latest occurrence of each code as a literature-aligned
  comparator. This is a separate baseline, not the primary interpretation of Jeff's comment.
- preserve deterministic event ordering and the correct final-window time label where time is used.

### 7. Strengthen integrity manifests

For every experimental instance, support private hashes of:

- dataset/config/model/tokenizer revision;
- task, canonical split, patient, and prediction instance;
- uncensored source events;
- post-deduplication events;
- retained content after packing;
- rendered text;
- temporal condition and perturbation seed.

The hashes must make silent content drift detectable without printing patient content.

## Slurm and environment reliability

Create `scripts/audit_environment.py` that reports, without absolute home paths:

- Python version and platform;
- import success and installed versions for all core, `dev`, `notebook`, and `model` dependencies;
- `pip check` outcome;
- PyTorch version, CUDA availability, visible device count, and device names when available;
- Transformers and tokenizer availability; and
- whether required environment variables point to existing logical assets.

This script must distinguish dependency groups *defined in `pyproject.toml`* from packages actually
installed in the active environment. Do not claim that reading `pyproject.toml` proves installation.

Create `scripts/slurm/11_run_phase1_alignment.sbatch` that:

- fails closed on missing project/data/asset/tokenizer/cohort paths;
- uses writable project-private `IPYTHONDIR`, `JUPYTER_CONFIG_DIR`, `XDG_CACHE_HOME`,
  `MPLCONFIGDIR`, and Hugging Face cache directories;
- avoids the `/etc/bash.bashrc` `PS1: unbound variable` failure mode;
- writes `RUNNING.json`, then exactly one of `COMPLETE.json` or `FAILED.json` under a private,
  job-specific status directory, including job ID, timestamps, Git commit, environment-audit path,
  notebook output paths, and exit status;
- executes only clean source notebooks and never overwrites them; and
- preserves the no-external-API and no-test-performance rules.

Create `notebooks/03_phase1_protocol_review.ipynb`. It should orchestrate tested modules and show
only privacy-safe aggregate results:

- protocol/config status and unresolved decisions;
- duplicate-removal counts;
- 30/90-day block distributions;
- eligibility for C2 and each C3 seed;
- content-matching success and retained-content counts at 4K and 8K;
- aggregate time-token cost and displacement under independent packing;
- latest-window and latest-code event-retention fractions; and
- a machine-readable go/no-go checklist for Phase 3.

Never display a real serialized history. Use a fully synthetic example to explain C0-C3.

Create `scripts/review_phase1_results.py` that reads only privacy-safe aggregate report files and
prints a concise status summary. It must not open the raw event parquet, cohort, predictions,
embeddings, or patient-level manifests.

## Testing requirements

Add synthetic tests for:

- exact prediction-relative window boundaries and time-of-day behavior;
- same-time concurrency;
- exact duplicate removal versus preserved recurrence;
- all C0-C3 invariants;
- C2 eligibility and gap reversal;
- three deterministic C3 seeds;
- shared-suffix content matching at 4K and 8K;
- independent-packing displacement accounting;
- latest-window and latest-code baselines;
- configuration status gates (`PILOT`, `PROPOSED`, `PENDING`, `LOCKED`);
- private versus aggregate output paths;
- clean source notebooks with required headings; and
- Slurm status-sentinel behavior where practical without a scheduler.

Update deprecated DuckDB Arrow-reader calls if the current installed DuckDB provides the supported
replacement. Preserve compatibility within the pinned dependency range and add tests.

## Privacy and test-set rules

- Do not print, log, display, or commit patient identifiers, timestamps, raw rows, real serialized
  histories, individual labels or predictions, or embeddings.
- Do not open governed patient-level artifacts merely to create examples.
- Do not use any external API.
- Do not run any test-set model evaluation. Aggregate split counts may be used only for integrity
  checks and must not guide design choices.
- Do not commit or push.

## Validation to run now

Run all validation that does not require an embedding-model forward pass:

```bash
python -m ruff check .
python -m pytest
bash -n scripts/slurm/10_run_phase1_notebooks.sbatch
bash -n scripts/slurm/11_run_phase1_alignment.sbatch
bash scripts/git/pre_push_safety.sh
```

If real EHRSHOT paths and the already-downloaded Qwen tokenizer are available, run only the
privacy-safe aggregate protocol-review notebook on train/validation. Otherwise give the exact DGX
command without inventing results.

## Completion report

Finish with:

1. a concise inventory of files changed;
2. tests and checks run with exact outcomes;
3. every protocol deviation found and whether it remains unresolved;
4. the exact DGX command to run notebook 03;
5. the exact aggregate files the user should inspect afterward; and
6. an explicit statement that no embedding experiment or test evaluation was run.
