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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


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
    source_commit = payload.get("source_git_commit")
    config_hash = payload.get("experiment_config_hash")
    provenance = payload.get("fit_input_provenance", {})
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or not _is_sha256(config_hash)
        or provenance.get("fit_source_git_commit") != source_commit
        or provenance.get("experiment_config_hash") != config_hash
        or provenance.get("expensive_stages_rerun") is not False
        or provenance.get("test_predictions_or_performance_used") is not False
    ):
        raise ContractError("Final M1 fit-input provenance is incomplete or inconsistent.")
    if provenance.get("mode") == "validated_fit_only_recovery":
        upstream = provenance.get("upstream_source_git_commit")
        source_jobs = provenance.get("source_jobs", {})
        foundation_job = source_jobs.get("foundation", {})
        throughput_job = source_jobs.get("throughput", {})
        embedding_job = source_jobs.get("embedding_array", {})
        failed_fit_job = source_jobs.get("failed_fit", {})
        if (
            upstream != "77c47f322b98fc74196cc56a0a16edc43604c887"
            or foundation_job.get("job_id") != 437
            or foundation_job.get("source_git_commit")
            != "1eb0d3535b0afaff0b82b4d430144ca7df372a8c"
            or not _is_sha256(foundation_job.get("complete_sentinel_sha256"))
            or throughput_job.get("job_id") != 444
            or throughput_job.get("source_git_commit") != upstream
            or not _is_sha256(throughput_job.get("complete_sentinel_sha256"))
            or embedding_job.get("job_id") != 445
            or embedding_job.get("indices") != [0, 1, 2, 3]
            or embedding_job.get("source_git_commit") != upstream
            or set(embedding_job.get("complete_sentinel_sha256_by_index", {}))
            != {"0", "1", "2", "3"}
            or any(
                not _is_sha256(value)
                for value in embedding_job.get("complete_sentinel_sha256_by_index", {}).values()
            )
            or failed_fit_job.get("job_id") != 446
            or failed_fit_job.get("source_git_commit") != upstream
            or not _is_sha256(failed_fit_job.get("failed_sentinel_sha256"))
            or failed_fit_job.get("preserved_unchanged") is not True
        ):
            raise ContractError("Final M1 fit-only recovery provenance drifted.")
    elif provenance.get("mode") != "same_source_downstream":
        raise ContractError("Final M1 fit-input provenance mode is not authorized.")
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
    provenance = payload["fit_input_provenance"]
    aggregate_hashes = provenance.get("aggregate_sha256", {})
    paths = {
        "foundation": root / "reports/generated/final/locked_four_task_m1_v1/foundation.json",
        "throughput": root / "reports/generated/final/locked_four_task_m1_v1/throughput.json",
    }
    if any(
        not path.is_file() or file_sha256(path) != aggregate_hashes.get(name)
        for name, path in paths.items()
    ):
        raise ContractError("Final fit source aggregate hashes drifted.")
    embedding_hashes = aggregate_hashes.get("embedding_by_task", {})
    embedding_paths = {
        task: root / f"reports/generated/final/locked_four_task_m1_v1/embedding/{task}.json"
        for task in FINAL_TASKS
    }
    if set(embedding_hashes) != set(FINAL_TASKS) or any(
        not embedding_paths[task].is_file()
        or file_sha256(embedding_paths[task]) != embedding_hashes[task]
        for task in FINAL_TASKS
    ):
        raise ContractError("Final fit source embedding aggregate hashes drifted.")
    if provenance.get("mode") == "validated_fit_only_recovery":
        failed_fit = provenance["source_jobs"]["failed_fit"]
        failed_path = root / f"outputs/private/run_status/{failed_fit['job_id']}/FAILED.json"
        if not failed_path.is_file() or file_sha256(failed_path) != failed_fit["failed_sentinel_sha256"]:
            raise ContractError("Source fit FAILED sentinel was not preserved unchanged.")


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
