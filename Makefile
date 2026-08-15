.PHONY: setup setup-model setup-notebook test lint audit validate-labels cohort censor safety

PYTHON ?= python
export PYTHONPATH := src
DATA_PATH ?= $${EHRSHOT_DATA_PATH}
ASSET_ROOT ?= $${EHRSHOT_ASSET_ROOT}

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

setup-model:
	$(PYTHON) -m pip install -e ".[dev,model]"

setup-notebook:
	$(PYTHON) -m pip install -e ".[dev,model,notebook]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

audit:
	$(PYTHON) -m temporal_ehr.cli audit --data-path "$(DATA_PATH)" --asset-root "$(ASSET_ROOT)" --output-dir reports/generated/data_audit

validate-labels:
	$(PYTHON) -m temporal_ehr.cli validate-labels --asset-root "$(ASSET_ROOT)" --output-dir reports/generated/label_validation

cohort:
	$(PYTHON) -m temporal_ehr.cli build-cohort --asset-root "$(ASSET_ROOT)" --allow-private-output

censor:
	$(PYTHON) -m temporal_ehr.cli validate-censoring --data-path "$(DATA_PATH)" --cohort-path outputs/private/cohorts/primary_panel.parquet

safety:
	bash scripts/git/pre_push_safety.sh
