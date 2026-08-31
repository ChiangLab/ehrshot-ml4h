#!/usr/bin/env python3
"""Semantic and privacy review of the final four-task M1 validation aggregate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import FINAL_TASKS
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import require_aggregate_output

FORBIDDEN_ROW_KEYS = {
    "subject_id",
    "patient_id",
    "instance_key",
    "prediction_time",
    "window_start",
    "window_end",
    "target",
    "probability",
    "text",
    "embedding",
    "embeddings",
}
SEEDS = (20260815, 20260816, 20260817)
METRICS = {"auroc", "auprc", "brier_score", "log_loss"}


def _reject_row_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_ROW_KEYS:
                raise ContractError(f"Final M1 aggregate contains prohibited row field: {key}")
            _reject_row_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_row_fields(item)


def _require_metrics(metrics: Any, context: str) -> None:
    if not isinstance(metrics, dict) or set(metrics) != METRICS:
        raise ContractError(f"{context} lacks the exact metric set.")
    if any(not isinstance(value, int | float) or not math.isfinite(value) for value in metrics.values()):
        raise ContractError(f"{context} contains non-finite metrics.")


def _require_fit(fit: Any, context: str) -> None:
    if not isinstance(fit, dict):
        raise ContractError(f"{context} is not a fit report.")
    _require_metrics(fit.get("metrics"), context)
    if fit.get("calibration", {}).get("status") not in {"REPORTED", "SUPPRESSED"}:
        raise ContractError(f"{context} lacks calibration.")


def _require_contrast(item: Any) -> None:
    if (
        not isinstance(item, dict)
        or not isinstance(item.get("estimate"), int | float)
        or len(item.get("ci_95", [])) != 2
        or any(not math.isfinite(float(value)) for value in [item["estimate"], *item["ci_95"]])
        or int(item.get("instances", 0)) < 1
        or int(item.get("valid_replicates", 0)) < 1
    ):
        raise ContractError("Final paired patient-bootstrap contrast is incomplete.")


def validate_final_m1_report(payload: dict[str, Any]) -> None:
    _reject_row_fields(payload)
    if payload.get("status") != "PASS" or payload.get("protocol_decision") != (
        "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED"
    ):
        raise ContractError("Final M1 report lacks its exact PASS/test-locked decision.")
    if any(
        payload.get(field) is not expected
        for field, expected in (
            ("content_matching_passed", True),
            ("label_index_only_perturbations_passed", True),
            ("patient_window_metadata_alignment_passed", True),
            ("pilot_artifacts_reused", False),
            ("test_predictions_or_performance_used", False),
            ("test_results_reported", False),
            ("m2_run", False),
            ("m3_run", False),
        )
    ):
        raise ContractError("Final M1 report violates content, isolation, test, or model-scope gates.")
    scope = payload.get("scientific_scope", {})
    if (
        scope.get("tasks") != list(FINAL_TASKS)
        or scope.get("splits") != ["train", "validation"]
        or scope.get("primary_model") != "Qwen/Qwen3-Embedding-0.6B"
        or scope.get("confirmatory_model") != "Qwen/Qwen3-Embedding-8B"
        or scope.get("token_budget") != 8192
        or scope.get("baselines") != ["B1", "B2", "B3", "B4"]
        or scope.get("conditions") != ["C0", "C1", "C2", "C3"]
        or scope.get("c3_seeds") != list(SEEDS)
    ):
        raise ContractError("Final M1 scientific scope drifted.")
    primary = payload.get("primary_model_identity", {})
    if (
        primary.get("frozen") is not True
        or primary.get("full_dimension") is not True
        or primary.get("l2_normalized_float32") is not True
    ):
        raise ContractError("Primary encoder is not frozen, full-dimensional, and FP32-normalized.")
    foundation = payload.get("foundation", {})
    if (
        foundation.get("status") != "PASS"
        or foundation.get("content_matching_passed") is not True
        or foundation.get("label_index_only_perturbations_passed") is not True
        or foundation.get("patient_window_metadata_alignment_passed") is not True
        or foundation.get("test_predictions_or_performance_used") is not False
        or {item.get("task") for item in foundation.get("task_summaries", [])} != set(FINAL_TASKS)
    ):
        raise ContractError("Final report lacks foundation history/recurrence/truncation/perturbation evidence.")
    for summary in foundation["task_summaries"]:
        if any(
            field not in summary
            for field in (
                "representation_coverage",
                "recurrent_event_count_after_exact_deduplication",
                "truncated_embedding_rows",
                "perturbation_strength",
            )
        ):
            raise ContractError("Final report lacks one required foundation coverage field.")
    throughput = payload.get("throughput", {})
    feasibility = throughput.get("feasibility_plan", {})
    foundation_provenance = throughput.get("foundation_provenance", {})
    if (
        throughput.get("status") != "PASS"
        or throughput.get("outcomes_or_performance_read") is not False
        or feasibility.get("contingency_multiplier") != 2.0
        or feasibility.get("primary_feasible_with_2x_contingency") is not True
        or feasibility.get("outcome_performance_used_for_feasibility") is not False
        or set(feasibility.get("tasks", {})) != set(FINAL_TASKS)
        or any(
            task.get("minimum_runtime_reserve_seconds") != 300
            for task in feasibility.get("tasks", {}).values()
        )
        or foundation_provenance.get("mode")
        not in {"same_source", "validated_execution_only_reuse"}
        or (
            foundation_provenance.get("mode") == "validated_execution_only_reuse"
            and (
                foundation_provenance.get("test_predictions_or_performance_used") is not False
                or foundation_provenance.get("cohort_or_prepared_artifact_changed") is not False
            )
        )
    ):
        raise ContractError("Final throughput/8B feasibility gate failed.")
    embedding_tasks = payload.get("embedding_tasks", [])
    if {item.get("task") for item in embedding_tasks} != set(FINAL_TASKS):
        raise ContractError("Final embedding reports do not cover exactly four tasks.")
    if any(
        item.get("status") != "PASS"
        or item.get("condition_level_caches") is not True
        or item.get("immutable_verified_cache_reuse") is not True
        or item.get("pilot_cache_reused") is not False
        or item.get("metadata_mirrors_every_embedding_row") is not True
        or item.get("test_split_used") is not False
        for item in embedding_tasks
    ):
        raise ContractError("One final task embedding report failed its cache/alignment gates.")
    tasks = payload.get("tasks", {})
    if set(tasks) != set(FINAL_TASKS):
        raise ContractError("Final M1 task reports do not cover exactly the locked panel.")
    required_scopes = {
        "complete_cohort",
        "b1_aligned",
        "b2_aligned",
        "c2_perturbable",
        *(f"c3_seed_{seed}" for seed in SEEDS),
    }
    for task, report in tasks.items():
        scopes = report.get("primary_scopes", {})
        if set(scopes) != required_scopes:
            raise ContractError(f"{task}: primary scopes are incomplete.")
        for scope_name, scope_report in scopes.items():
            if int(scope_report.get("train_instances", 0)) < 1 or int(scope_report.get("validation_instances", 0)) < 1:
                raise ContractError(f"{task}/{scope_name}: empty fitting scope.")
            for label, fit in scope_report.get("representations", {}).items():
                _require_fit(fit, f"{task}/{scope_name}/{label}")
        _require_metrics(report.get("B3_code_count", {}).get("metrics", {}).get("model"), f"{task}/B3")
        _require_metrics(report.get("B4_training_prevalence", {}).get("metrics"), f"{task}/B4")
        if len(report.get("paired_patient_bootstrap_contrasts", [])) != 6:
            raise ContractError(f"{task}: primary paired contrasts are incomplete.")
        for item in report["paired_patient_bootstrap_contrasts"]:
            _require_contrast(item)
        if len(report.get("c3_seed_contrasts", [])) != 3:
            raise ContractError(f"{task}: C3 seed contrasts are incomplete.")
        for item in report["c3_seed_contrasts"]:
            _require_contrast(item)
        if report.get("c3_seed_variability", {}).get("seeds") != list(SEEDS):
            raise ContractError(f"{task}: C3 seed variability is incomplete.")
    macro = payload.get("equal_weight_panel_macro", {})
    expected_macro = {
        "C0_complete_cohort",
        "C1_complete_cohort",
        "B1_latest_window",
        "B2_latest_code",
        "C2_perturbable",
        "B3_code_count",
        "B4_training_prevalence",
        *(f"C3_seed_{seed}_perturbable" for seed in SEEDS),
    }
    if set(macro) != expected_macro:
        raise ContractError("Equal-weight panel macro estimates are incomplete.")
    for name, estimate in macro.items():
        if estimate.get("task_weighting") != "equal" or estimate.get("task_count") != 4:
            raise ContractError(f"{name}: macro weighting drifted.")
        _require_metrics(estimate.get("metrics"), name)
    macro_contrasts = payload.get("equal_weight_panel_macro_paired_bootstrap_contrasts", [])
    if len(macro_contrasts) != 7:
        raise ContractError("Equal-weight macro paired contrasts are incomplete.")
    if any(
        int(item.get("valid_replicates", 0)) < 1
        or len(item.get("ci_95", [])) != 2
        or not math.isfinite(float(item.get("estimate", float("nan"))))
        for item in macro_contrasts
    ):
        raise ContractError("One equal-weight macro paired contrast failed.")
    if not isinstance(payload.get("private_prediction_rows"), int) or payload["private_prediction_rows"] < 1:
        raise ContractError("Final M1 report lacks private prediction count/hash evidence.")
    if (
        not isinstance(payload.get("private_predictions_sha256"), str)
        or len(payload["private_predictions_sha256"]) != 64
    ):
        raise ContractError("Final M1 report lacks a private prediction SHA-256.")


def validate_private_postconditions(payload: dict[str, Any], root: Path) -> None:
    predictions = root / "outputs/private/final/locked_four_task_m1_v1/fit/validation_predictions.parquet"
    if not predictions.is_file() or file_sha256(predictions) != payload.get("private_predictions_sha256"):
        raise ContractError("Final private predictions are missing or hash-mismatched.")
    table = pq.read_table(predictions, columns=["task", "scope"])
    if len(table) != payload.get("private_prediction_rows"):
        raise ContractError("Final private prediction row count drifted.")
    if set(table["task"].to_pylist()) != set(FINAL_TASKS):
        raise ContractError("Final private predictions do not cover exactly four tasks.")


def load_final_m1_report(path: Path, root: Path, *, private_postconditions: bool) -> dict[str, Any]:
    destination = require_aggregate_output(path, root)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert_privacy_safe_report(json.dumps(payload, sort_keys=True))
    validate_final_m1_report(payload)
    if private_postconditions:
        validate_private_postconditions(payload, root)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--private-postconditions", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = args.report if args.report.is_absolute() else root / args.report
    payload = load_final_m1_report(report, root, private_postconditions=args.private_postconditions)
    print(f"PASS: reviewed final M1 validation for {len(payload['tasks'])} tasks; test remains locked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
