# Temporal EHR ML4H

This repository implements **Does Time Earn Its Tokens? Recurrence, Temporal Alignment, and
Context Cost in Frozen Longitudinal EHR Embeddings**. The current code audits the real downloaded
EHRSHOT event table, validates all 15 task-label files and canonical patient splits, selects a
patient-balanced three-task pilot cohort, audits strict prediction-time censoring, and implements
protocol-aligned prediction-relative serialization before any embedding experiment.

Repository code is MIT-licensed. That license does not apply to EHRSHOT data or other governed
assets, which remain subject to their official DUA and redistribution restrictions.

The raw Parquet file is queried in place with DuckDB. It is never copied into this repository or
converted to a persistent database. Governed cohorts, model weights, embeddings, and generated
reports remain Git-ignored. Unit tests use synthetic fixtures, but all scientific validation and
later experiments use the real EHRSHOT release.

## Setup

Use a modern supported Python (3.11 or later) and a local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The checked-in `.env.example` contains the project-neutral path requested by the protocol. In the
current workspace the files were discovered under the sibling `data/` directory, so either pass
the paths explicitly or export the local relative overrides:

```bash
export EHRSHOT_DATA_PATH=../data/data.parquet
export EHRSHOT_ASSET_ROOT=../data
```

The Slurm scripts contain overridable defaults for the assigned MedNet project allocation. Local
Mac paths are supplied through environment variables and never written to reports.

## Run the safe asset inventory

This command recursively inspects filenames and file types, ZIP member names, checksums, safe
version metadata, and Parquet schemas. It never displays patient rows or identifiers.

```bash
python -m temporal_ehr.cli inventory \
  --data-path ../data/data.parquet \
  --asset-root ../data \
  --output-dir reports/generated/data_audit
```

## Run the complete real-data setup

```bash
bash scripts/run_real_ehrshot_setup.sh
```

Individual commands remain available:

```bash
make audit
make validate-labels
make cohort
```

The commands write:

- `reports/generated/data_audit/audit.json` and `audit.md`;
- `reports/generated/data_audit/asset_inventory.json` and `asset_inventory.md`;
- five figures and adjacent aggregate CSV plot tables under `figures/`;
- `reports/generated/NEXT_ASSETS_REQUIRED.md`.
- `reports/generated/label_validation/label_validation.{json,md}`;
- `reports/generated/cohort/cohort_summary.{json,md}`;
- governed `outputs/private/cohorts/primary_panel.parquet`.

All of these paths are ignored by Git. The CLI rejects report text containing scalar
subject/patient identifiers or absolute user-home paths before writing it.

## Verify

```bash
python -m pytest
python -m ruff check .
python -m temporal_ehr.cli validate-labels --asset-root ../data
python -m temporal_ehr.cli build-cohort --asset-root ../data --allow-private-output
git check-ignore reports/generated/data_audit/audit.json
bash scripts/git/pre_push_safety.sh
```

Unit tests use entirely synthetic event tables with three invented subjects, multi-year and irregular
gaps, concurrent timestamps, a deliberate exact duplicate, recurrent codes, optional values,
person and death events, and multiple same-day events. No real EHRSHOT row appears in the suite.

## Scientific interpretation limits

- A `visit_occurrence` row is not necessarily a true encounter.
- A calendar-day block is a deterministic fallback, not an encounter identifier.
- Identical timestamps indicate recorded concurrency, not duplication.
- Exact duplicates match across all seven stored columns.
- Repeated codes across dates are recurrence and must not be treated as data errors.
- `note` source rows without `text_value` do not constitute narrative clinical notes.

Phase 1 provides a simple count-based reference followed by frozen-serializer QC. It is not labeled
as a released EHRSHOT baseline reproduction. The test split remains untouched until the
representation, model, and statistical-analysis plan are frozen.

The primary serializer uses exact half-open prediction-relative 30/90-day windows, removes only
seven-field exact duplicates, and renders range labels without calendar dates. See
[the serialization protocol](docs/SERIALIZATION_PROTOCOL.md) and
[the deviation register](docs/PROTOCOL_DEVIATIONS.md).

## Phase 1 notebook pack

Install the headless notebook, modeling, and local-tokenizer dependencies without downloading any
model weights:

```bash
python -m pip install -e ".[dev,model,notebook]"
```

The clean notebooks under `notebooks/` contain orchestration and aggregate displays only. Reusable
scientific logic lives under `src/temporal_ehr/`. Execute a single clean source notebook into the
Git-ignored private output tree with:

```bash
python scripts/execute_notebook.py \
  notebooks/00_ehrshot_foundation.ipynb \
  outputs/private/executed_notebooks/00_ehrshot_foundation.ipynb
```

On the DGX, after the governed cohort and pinned local Qwen tokenizer are present, execute all three
in order:

```bash
export QWEN_EMBEDDING_PATH="${TEMPORAL_EHR_MODEL_ROOT}/Qwen3-Embedding-8B"
sbatch --export=ALL scripts/slurm/10_run_phase1_notebooks.sbatch
```

The runner never overwrites source notebooks. Count matrices, models, predictions, executed
notebooks, and generated aggregate reports remain ignored pending review. Notebook model selection
fails closed on any request for the test split.

## Phase 2 protocol review

The existing three-task cohort remains explicitly marked `PILOT`; the four-task written-protocol
panel remains `PROPOSED`; and `configs/experiments/locked_panel.yaml` remains `PENDING`. Downstream
embedding training must call the lock gate and cannot proceed until one reviewed panel is marked
`LOCKED` after confirmation with Jeff.

On the DGX, run only the privacy-safe aggregate review notebook with the local tokenizer (no model
forward pass):

```bash
export EHRSHOT_COHORT_PATH="$PWD/outputs/private/cohorts/primary_panel.parquet"
sbatch --export=ALL scripts/slurm/11_run_phase1_alignment.sbatch
```

The job creates a private environment audit, hash-only instance manifests, an executed Notebook 03,
and atomic job-status sentinels. Inspect only the aggregate report with:

```bash
.venv/bin/python scripts/review_phase1_results.py
```

## Execution and governance

- [August 14–16 runbook](docs/AUG14_16_RUNBOOK.md)
- [Task-selection review](docs/TASK_SELECTION_REVIEW.md)
- `scripts/git/pre_push_safety.sh` blocks common governed-data/model formats and obvious secrets.
- `scripts/git/configure_mednet_remote.sh` configures the exact private MedNet Git remote only after
  a verified repository URL and author email are supplied.
- DGX bootstrap, model download, smoke tests, and censoring validation are under `scripts/slurm/`.
