# Codex prompt: build the first DGX-ready notebook pack

Work in the current `temporal-ehr-ml4h` repository. Read the repository before editing, especially:

- `README.md`
- `configs/expected_ehrshot.yaml`
- `configs/tasks.yaml`
- `docs/AUG14_16_RUNBOOK.md`
- `docs/TASK_SELECTION_REVIEW.md`
- `src/temporal_ehr/data/`
- `tests/`
- `.gitignore`

## Goal

Implement the first reproducible, headlessly executable notebook pack for the temporal EHR study.
The notebooks will be committed to the private MedNet Git repository and executed later on a DGX
Spark machine. Keep notebooks thin: scientific logic belongs in importable Python modules with
synthetic unit tests, while notebooks should orchestrate the modules, explain decisions, and show
aggregate results.

Create exactly these source notebooks:

1. `notebooks/00_ehrshot_foundation.ipynb`
2. `notebooks/01_count_baseline.ipynb`
3. `notebooks/02_serialization_qc.ipynb`

Also create a Slurm runner that executes clean copies in that order:

- `scripts/slurm/10_run_phase1_notebooks.sbatch`

Executed notebooks must be written under `outputs/private/executed_notebooks/`; never overwrite the
clean source notebooks. Add any necessary notebook dependencies as a `notebook` optional dependency
group in `pyproject.toml` and provide a deterministic headless execution script.

## Governing scientific decisions

- Use the real downloaded EHRSHOT release for scientific computation. Synthetic fixtures are only
  for unit tests and visibly labeled demonstrations.
- The primary task panel is `lab_anemia`, `new_acutemi`, and `guo_los`.
- Use the canonical patient splits. During this phase, allow only `train` and `validation` for
  modeling and model selection. Code must fail closed if a notebook requests `test`.
- Use the existing governed cohort at
  `outputs/private/cohorts/primary_panel.parquet`. If it is absent, print the exact existing CLI
  command needed to rebuild it; do not silently create a different cohort.
- Every history must use strict censoring: `event_time < prediction_time`. Events at exactly the
  prediction timestamp must be excluded.
- EHRSHOT has no reliable encounter identifier in this MEDS conversion. Treat a calendar day as the
  atomic clinical block. Support optional aggregation of day blocks into prediction-time-anchored
  30-day and 90-day windows for QC, but do not silently declare either window size the final primary
  setting.
- Same-time events are concurrent. Sort them deterministically without implying clinical order.
- Do not infer narrative notes; the downloaded `note` rows have no narrative `text_value`.
- The project is predictive, not causal. Do not use causal language.

## Privacy and governance constraints

- Do not display, print, log, or commit patient identifiers, patient-level timestamps, raw event
  rows, serialized real-patient histories, individual predictions, or embeddings.
- Do not read governed files merely to show examples. Use synthetic examples in notebook displays.
- Real-data notebook outputs must be aggregate tables or aggregate figures with minimum group size
  20. Suppress smaller groups.
- Do not send data to any external API. Do not use the existing local 32B chat model in this task.
- All patient-level feature matrices, cohorts, caches, serialized histories, and predictions must go
  under `outputs/private/` and remain Git-ignored.
- Do not download embedding models and do not implement the Qwen embedding experiment in this task.
- Do not commit or push. Leave changes uncommitted for review.

## Notebook 00: trusted foundation

Purpose: make the validated data foundation easy to rerun and inspect on the DGX without exposing
patient rows.

Requirements:

- Resolve `EHRSHOT_DATA_PATH` and `EHRSHOT_ASSET_ROOT` only from environment variables.
- Display environment/package versions, logical filenames, and the pinned configuration—not
  absolute user-home paths.
- Call or import the existing event audit, label validation, cohort-summary, and strict censoring
  code. Do not duplicate those implementations in notebook cells.
- Show aggregate task counts and split counts for the three primary tasks.
- Show the prevalence difference between all source label rows and the one-row-per-patient cohort,
  explicitly explaining that frequent testers otherwise receive more weight.
- Show aggregate counts of exact-prediction-time events excluded by strict censoring.
- End with a machine-readable checklist stating whether the repository is ready for the count
  baseline and serialization QC.

## Notebook 01: reproducible count baseline

Purpose: establish a simple, defensible non-embedding reference before testing text embeddings.

Implement reusable code under appropriate modules such as:

- `src/temporal_ehr/features/count_baseline.py`
- `src/temporal_ehr/evaluation/metrics.py`
- `src/temporal_ehr/data/history.py`

Requirements:

- Extract only events strictly before each prediction time.
- Build sparse per-instance counts of structured EHR codes. Do not load the 41.7-million-row event
  table into pandas or Python lists. Use DuckDB/Arrow streaming and sparse matrices.
- Fit the feature vocabulary on training data only. Apply `log1p` to nonzero counts as a fixed,
  documented transform.
- Fit a separate scikit-learn logistic regression for each task.
- Use a small explicit regularization grid, selected on validation AUROC. Record every candidate and
  the deterministic tie rule. Do not inspect test performance.
- Report validation AUROC, AUPRC, log loss, and Brier score with prevalence and a prevalence-only
  reference. Label all results as validation estimates.
- Save private matrices/models/predictions under `outputs/private/count_baseline/`.
- Save a privacy-safe aggregate JSON result under `reports/generated/count_baseline/`; keep it
  Git-ignored pending review.
- Add checks that input patients do not cross canonical splits and that no event at or after the
  prediction time enters the features.

This is a baseline, not the final model and not a released EHRSHOT reproduction unless every detail
matches the official released baseline. Use the phrase “simple count-based reference” unless such
reproduction is actually demonstrated.

## Notebook 02: serialization QC before embeddings

Purpose: validate the longitudinal text construction and temporal perturbations before paying for
embedding computation.

Implement reusable code under an explicit package such as:

- `src/temporal_ehr/serialization/blocks.py`
- `src/temporal_ehr/serialization/conditions.py`
- `src/temporal_ehr/serialization/render.py`
- `src/temporal_ehr/serialization/packing.py`

Requirements:

1. Convert code values to official descriptions using the downloaded code metadata, with a stable
   fallback to the code string.
2. Construct atomic calendar-day blocks from strictly pre-prediction events.
3. Preserve recurrent events in the full-history representation.
4. Implement these full-history temporal conditions without changing clinical event content:
   - `order_only`: chronological events, no elapsed-time tokens.
   - `correct_compact_time`: correct relative-time labels.
   - `reversed_earlier_gaps`: reverse the sequence of earlier inter-block gaps while keeping the
     final block’s time label and all event order fixed.
   - `shuffled_earlier_time_labels`: deterministically permute only earlier time labels while keeping
     the final block’s time label and all event order fixed.
5. Implement a separate `latest_window_only` information-reduction baseline. It is not a second row
   of a 2-by-4 matrix and must not be described as preserving the complete trajectory.
6. Support fixed perturbation seeds and assert:
   - identical patients, outcomes, event content, and event order across the four full-history
     conditions;
   - only the intended time-label mapping changes;
   - the final time label remains correct in both falsified-time conditions;
   - all falsified labels remain within the observed history span;
   - histories with too few blocks are marked ineligible rather than fabricated.
7. Provide a deterministic token-budget packer with a configurable primary budget of 4,096 Qwen
   tokens. If the Qwen tokenizer is unavailable during local development, fail with a clear message
   for real-data packing; a whitespace estimate may be shown only for synthetic tests and must not
   be labeled a token count.
8. Do not print real serialized histories. Show the four conditions using one entirely synthetic
   timeline, then show only aggregate real-data distributions: event counts, block counts, history
   span, token counts when available, truncation rate, and clinical events displaced by time tokens.
9. Compare calendar-day atomic blocks with 30-day and 90-day aggregated windows using aggregate QC
   only. Do not choose the final window based on test data.

## Testing and implementation standards

- Extend the existing synthetic fixtures to cover concurrency, recurrent codes, one-block and
  two-block histories, long gaps, exact prediction-time events, and truncation.
- Test every temporal-condition invariant directly.
- Test deterministic output under repeated runs and fixed seeds.
- Test that source notebooks contain no outputs before commit.
- Add a lightweight notebook structure test that verifies required headings and cell order.
- Use functions and configuration rather than hidden notebook state.
- Do not hardcode Mac, MedNet, or DGX data paths in Python.
- Preserve existing working code unless a change is necessary and tested.
- Use `apply_patch` for edits.

## Validation to run now

Run all validation that does not require a GPU or external download:

```bash
python -m ruff check .
python -m pytest
bash -n scripts/slurm/10_run_phase1_notebooks.sbatch
bash scripts/git/pre_push_safety.sh
```

If the local real EHRSHOT paths are available, run only privacy-safe aggregate QC and the count
baseline on train/validation. Do not run or reveal test metrics. If dependencies or data are absent,
do not invent results; report the exact DGX commands that remain.

## Completion criteria

- All three clean source notebooks exist and contain no stored output.
- Scientific logic lives in tested modules, not duplicated notebook cells.
- The count baseline can run headlessly using train/validation only.
- All four temporal conditions pass content/order/final-time invariants on synthetic fixtures.
- Real histories are never displayed or committed.
- The Slurm runner writes executed notebooks and private artifacts only to ignored paths.
- Existing tests plus new tests pass; Ruff, shell syntax, and the pre-push safety gate pass.
- Finish with a concise report of files created, commands run, tests passed, and DGX-only work left.
