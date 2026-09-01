#!/usr/bin/env python3
"""Semantic/privacy validator for Prompt 4 aggregate reports and frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import FINAL_TASKS
from temporal_ehr.serialization.integrity import stable_hash

FORBIDDEN_KEYS = {
    "subject_id",
    "patient_id",
    "instance_key",
    "prediction_time",
    "target",
    "probability",
    "summary",
    "text",
    "embedding",
    "embeddings",
}


def _reject_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_KEYS:
                raise ContractError(f"Prompt 4 aggregate contains prohibited row field: {key}")
            _reject_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_fields(item)


def _metrics(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"auroc", "auprc", "brier_score", "log_loss"}:
        raise ContractError("Prompt 4 aggregate lacks the exact validation metric set.")


def validate_prompt4_aggregates(root: Path, *, require_frozen: bool) -> None:
    aggregate = root / "reports/generated/final/locked_four_task_m2_m3_v1"
    m2 = json.loads((aggregate / "m2_reference.json").read_text(encoding="utf-8"))
    m3 = json.loads((aggregate / "m3_reference.json").read_text(encoding="utf-8"))
    selection = json.loads((aggregate / "approach_selection.json").read_text(encoding="utf-8"))
    for payload in (m2, m3, selection):
        _reject_fields(payload)
        assert_privacy_safe_report(json.dumps(payload, sort_keys=True))
        if payload.get("status") != "PASS" or payload.get("test_predictions_or_performance_used") is not False:
            raise ContractError("Prompt 4 reference report failed or accessed held-out performance.")
    if set(m2.get("tasks", {})) != set(FINAL_TASKS) or set(m3.get("tasks", {})) != set(FINAL_TASKS):
        raise ContractError("Prompt 4 reference reports do not cover exactly four locked tasks.")
    _metrics(m2.get("equal_weight_panel_macro"))
    _metrics(m3.get("equal_weight_panel_macro"))
    fidelity = m2.get("fidelity_validation", {})
    if (
        fidelity.get("sample_histories") < 40
        or fidelity.get("task_condition_groups") != 8
        or fidelity.get("validation_only") is not True
        or fidelity.get("held_out_prompt_refinement") is not False
    ):
        raise ContractError("M2 fidelity validation is incomplete or not validation-only.")
    if m3.get("training_seeds") != [20260815, 20260816, 20260817]:
        raise ContractError("M3 does not report all three frozen training seeds.")
    if m3.get("mean_pool_logistic_baseline_reported") is not True:
        raise ContractError("M3 lacks its mean-pool logistic capacity baseline.")
    if (
        selection.get("reference_cells_passed") is not True
        or selection.get("selected_approach") not in {"M1", "M2", "M3"}
        or selection.get("minimum_margin") != 0.005
        or selection.get("no_margin_priority") != ["M1", "M2", "M3"]
        or selection.get("second_tiebreaker") != "validation log loss"
        or selection.get("test_results_reported") is not False
    ):
        raise ContractError("Prompt 4 approach selection drifted from its prespecification.")
    if not require_frozen:
        return
    manifest = json.loads((aggregate / "FROZEN_EXPERIMENT_MANIFEST.json").read_text(encoding="utf-8"))
    _reject_fields(manifest)
    digest = manifest.pop("frozen_manifest_sha256", None)
    if (
        digest != stable_hash(manifest)
        or manifest.get("selected_approach") != selection["selected_approach"]
        or manifest.get("protocol_frozen") is not True
        or manifest.get("one_time_held_out_run_authorized_by_prompt4") is not True
        or manifest.get("test_split_accessed") is not False
    ):
        raise ContractError("Prompt 4 frozen manifest hash or authorization is invalid.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    validate_prompt4_aggregates(root, require_frozen=args.require_frozen)
    print("PASS: Prompt 4 aggregate semantics, privacy, selection, and test lock")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
