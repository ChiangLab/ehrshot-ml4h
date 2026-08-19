# Temporal EHR project status and review guide

Status date: 2026-08-19

## Current conclusion

The repository has a sound, privacy-conscious Phase 1 implementation and a useful three-task pilot.
It is not ready for the primary embedding matrix until the final task panel is locked and several
serializer details are aligned with the final protocol.

## Completed

- EHRSHOT event-table audit and schema checks
- label and canonical split validation
- deterministic one-row-per-patient cohort for the current three-task pilot
- strict pre-prediction censoring
- three clean, headlessly executable source notebooks
- prevalence and count-based validation references for the pilot
- serializer and temporal-condition prototype
- DGX execution of notebooks 00, 01, and 02 to private output files
- 40 passing tests, Ruff pass, shell syntax pass, and pre-push safety pass at the last local review
- clean Git working tree at commit `bc5a746`, synchronized with `origin/main` at the last review

## Current validation-only pilot results

These are simple count-based references, not frozen-embedding results and not test-set results.

| Task | Validation AUROC | Validation AUPRC | Brier score | Log loss |
|---|---:|---:|---:|---:|
| Long length of stay (`guo_los`) | 0.756 | 0.496 | 0.158 | 0.535 |
| Anemia (`lab_anemia`) | 0.820 | 0.799 | 0.173 | 0.570 |
| New acute MI (`new_acutemi`) | 0.687 | 0.180 | 0.075 | 0.278 |

The acute-MI model ranks patients better than chance, but its Brier score was slightly worse than
the prevalence-only reference. This suggests useful ranking signal with weak probability
calibration; it is not a reason by itself to change the task.

## Not yet complete

- formal verification of DGX job 404's scheduler state and exit code
- retrieval/review of notebook 02's aggregate serialization-QC report from the DGX
- agreement on the final task panel
- protocol-aligned 30/90-day primary blocks
- exact duplicate removal
- explicit compact-time range labels
- primary content-matched packing across C0-C3
- three fixed C3 perturbation seeds and strengthened C2/C3 invariants
- latest-window and latest-code embedding baselines
- Qwen embedding throughput benchmark
- any Qwen embedding generation or embedding-model results
- M1 full temporal matrix, M2 interim-summary model, or M3 per-window sequence model
- frozen test-set evaluation

## Important protocol decision for Jeff

The current code uses the pilot panel:

- anemia
- new acute myocardial infarction
- long length of stay

The final written protocol proposed:

- anemia
- 30-day readmission
- new hypertension
- abnormal chest radiograph

The code should support either panel, but only one should be marked `LOCKED` before embedding runs.
Present all current metrics as results from the **three-task pilot**, not from the final task panel.

## Verify DGX job 404 after reconnecting

From the DGX login shell:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h

sacct -j 404 \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocCPUS

tail -n 200 slurm-tehr-phase1-notebooks-404.out

for notebook in 00_ehrshot_foundation 01_count_baseline 02_serialization_qc; do
  test -s "outputs/private/executed_notebooks/${notebook}.ipynb" \
    && echo "PRESENT ${notebook}" \
    || echo "MISSING ${notebook}"
done

.venv/bin/python -m pip check
```

An absent `output_type: error` in the executed notebooks is encouraging, but it does not replace a
`COMPLETED` scheduler state with exit code `0:0` and a clean final log.

## Verify what is actually installed

`pyproject.toml` defines dependency groups; it cannot prove which ones are installed in the active
DGX virtual environment. The earlier local-LLM answer accurately transcribed most definitions, but
incorrectly implied there was no conventional `dev` group and did not verify installation.

Run:

```bash
cd /data/shared73/projects/wdelery/temporal-ehr-ml4h

.venv/bin/python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

packages = [
    "duckdb", "matplotlib", "numpy", "pyarrow", "PyYAML",
    "joblib", "pytest", "ruff", "scikit-learn", "scipy",
    "ipykernel", "ipywidgets", "nbclient", "nbformat",
    "accelerate", "huggingface-hub", "sentence-transformers", "transformers",
    "torch",
]
for package in packages:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
PY

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## What to review while Codex implements the next stage

Review these in order:

1. `reports/generated/count_baseline/validation_results.json`: confirm the three validation-only
   results and selected regularization values.
2. `reports/generated/serialization_qc/serialization_qc.json` on the DGX: confirm it exists, then
   inspect only its aggregate fields.
3. `configs/tasks.yaml`: compare the current pilot panel with the proposed final panel.
4. `src/temporal_ehr/serialization/`: check whether blocks are day-based or task-family windows and
   whether packing is content matched.
5. `docs/PROTOCOL_DEVIATIONS.md` after Prompt 02: every design difference should be visible rather
   than silently fixed.

Questions to resolve with Jeff:

- Is the three-task panel the formal submission panel or only a pilot?
- If a four-task breadth panel is preferred, is the exact choice anemia, readmission, hypertension,
  and CheXpert?
- Are 30-day windows for lab/imaging/operational tasks and 90-day windows for incident diagnoses
  acceptable as the locked primary aggregation?
- Does Jeff agree that `latest_window_only` is a separate information-reduction baseline rather
  than a second row of the temporal perturbation matrix?
- Is the universal strict pre-prediction boundary acceptable as a conservative project rule, or
  should any task use a benchmark-specific boundary?

## Next sequence

1. Give Codex `prompts/02_AUDIT_PHASE1_AND_ALIGN_PROTOCOL.md`.
2. Review its diff, tests, deviation log, and aggregate notebook-03 output.
3. Resolve the task panel with Jeff and mark exactly one configuration `LOCKED`.
4. Give Codex `prompts/03_BUILD_PHASE3_BASELINES.md`.
5. After verified baseline completion, give Codex
   `prompts/04_BUILD_THROUGHPUT_AND_M1_PILOT.md`.

Do not skip directly to the 8B model or the full temporal matrix. Correctly freezing the content,
panel, windows, and baselines is what makes the later result interpretable.
