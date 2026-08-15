#!/usr/bin/env bash

set -euo pipefail

EVENT_PATH="${EHRSHOT_DATA_PATH:?Set EHRSHOT_DATA_PATH to the real EHRSHOT event Parquet.}"
ASSET_ROOT="${EHRSHOT_ASSET_ROOT:?Set EHRSHOT_ASSET_ROOT to the downloaded EHRSHOT asset directory.}"
PYTHON_BIN="${TEMPORAL_EHR_PYTHON:-.venv/bin/python}"
export PYTHONPATH="${PYTHONPATH:-src}"

"${PYTHON_BIN}" -m temporal_ehr.cli audit \
  --data-path "${EVENT_PATH}" \
  --asset-root "${ASSET_ROOT}" \
  --output-dir reports/generated/data_audit

"${PYTHON_BIN}" -m temporal_ehr.cli validate-labels \
  --asset-root "${ASSET_ROOT}" \
  --output-dir reports/generated/label_validation

"${PYTHON_BIN}" -m temporal_ehr.cli build-cohort \
  --asset-root "${ASSET_ROOT}" \
  --tasks lab_anemia new_acutemi guo_los \
  --selection-seed 20260814 \
  --private-output outputs/private/cohorts/primary_panel.parquet \
  --report-dir reports/generated/cohort \
  --allow-private-output

"${PYTHON_BIN}" -m pytest
"${PYTHON_BIN}" -m ruff check .

echo "Real-data setup complete. Run validate-censoring on the DGX before serialization."
