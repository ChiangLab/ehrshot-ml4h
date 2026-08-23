#!/usr/bin/env python3
"""Review only the privacy-safe complete-pilot aggregate report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import require_aggregate_output

PILOT_TASKS = ("lab_anemia", "new_acutemi", "guo_los")
PILOT_SEEDS = (20260815, 20260816, 20260817)
FORBIDDEN_ROW_KEYS = {
    "subject_id",
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


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_metrics(metrics: Any, context: str) -> None:
    required = {"auroc", "auprc", "brier_score", "log_loss"}
    if not isinstance(metrics, dict) or set(metrics) != required:
        raise ContractError(f"{context} lacks the exact required metric set.")
    if any(not isinstance(value, int | float) or not math.isfinite(value) for value in metrics.values()):
        raise ContractError(f"{context} contains a non-finite metric.")


def _require_calibration(value: Any, context: str) -> None:
    if not isinstance(value, dict) or value.get("status") not in {"REPORTED", "SUPPRESSED"}:
        raise ContractError(f"{context} lacks a valid calibration summary.")


def _reject_row_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_ROW_KEYS:
                raise ContractError(f"Pilot aggregate contains prohibited row-level field: {key}")
            _reject_row_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_row_fields(item)


def validate_pilot_report(payload: dict[str, Any]) -> None:
    """Require every scientific and operational pilot postcondition, not mere file existence."""
    _reject_row_fields(payload)
    if payload.get("status") != "PASS" or payload.get("protocol_decision") != (
        "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED"
    ):
        raise ContractError("Pilot aggregate does not record the required PASS decision.")
    if not _is_lower_hex(payload.get("experiment_config_hash"), 64) or not _is_lower_hex(
        payload.get("source_git_commit"), 40
    ):
        raise ContractError("Pilot aggregate lacks versioned configuration/source provenance.")
    scope = payload.get("scientific_scope", {})
    expected_scope = {
        "tasks": list(PILOT_TASKS),
        "splits": ["train", "validation"],
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "token_budget": 4096,
        "approach": "M1 direct serialization only",
        "baselines": ["B4", "B3", "B1", "B2"],
        "conditions": ["C0", "C1", "C2", "C3"],
        "c3_seeds": list(PILOT_SEEDS),
    }
    if not isinstance(scope, dict) or any(scope.get(key) != value for key, value in expected_scope.items()):
        raise ContractError("Pilot aggregate scientific scope drifted from the authorization.")
    if any(
        not _is_lower_hex(scope.get(field), 64)
        for field in ("model_revision_hash", "model_artifact_hash", "tokenizer_hash")
    ):
        raise ContractError("Pilot aggregate lacks pinned model/tokenizer hashes.")
    if any(
        bool(payload.get(field, True))
        for field in (
            "final_panel_used_for_pilot",
            "test_predictions_or_performance_used",
            "m2_run",
            "m3_run",
            "qwen_8b_run",
        )
    ):
        raise ContractError("Pilot aggregate violates a final-panel, test, M2/M3, or 8B prohibition.")
    if payload.get("content_matching_passed") is not True:
        raise ContractError("Pilot aggregate did not pass content matching.")
    prediction_rows = payload.get("private_prediction_rows")
    if (
        not _is_lower_hex(payload.get("private_predictions_sha256"), 64)
        or not isinstance(prediction_rows, int)
        or isinstance(prediction_rows, bool)
        or prediction_rows < 1
    ):
        raise ContractError("Pilot aggregate lacks private prediction count/hash evidence.")

    prepare = payload.get("prepare", {})
    summaries = prepare.get("task_summaries", []) if isinstance(prepare, dict) else []
    if prepare.get("status") != "PASS" or prepare.get("test_predictions_or_performance_used") is not False:
        raise ContractError("Pilot preparation postconditions did not pass.")
    if {summary.get("task") for summary in summaries} != set(PILOT_TASKS):
        raise ContractError("Pilot preparation summaries do not cover exactly the three pilot tasks.")
    for summary in summaries:
        if summary.get("content_matching_passed") is not True or summary.get("test_split_used") is not False:
            raise ContractError("A task preparation summary failed content or split controls.")
        if not isinstance(summary.get("representation_coverage"), dict):
            raise ContractError("A task preparation summary lacks representation coverage.")
        coverage = summary["representation_coverage"]
        if "C1_correct_compact_time" not in coverage:
            raise ContractError("A task preparation summary lacks complete-history coverage.")
        if any(
            field not in coverage["C1_correct_compact_time"]
            for field in ("eligible_histories", "retained_windows", "retained_events", "token_count")
        ):
            raise ContractError("A task preparation summary lacks required packing coverage fields.")

    throughput = payload.get("throughput", {})
    if (
        throughput.get("status") != "PASS"
        or throughput.get("failures") != 0
        or throughput.get("retries") != 0
        or throughput.get("test_outcomes_used") is not False
    ):
        raise ContractError("Pilot throughput postconditions did not pass.")
    embedding_tasks = payload.get("embedding_tasks", [])
    if {item.get("task") for item in embedding_tasks} != set(PILOT_TASKS):
        raise ContractError("Pilot embedding summaries do not cover exactly the three tasks.")
    if any(
        item.get("status") != "PASS"
        or item.get("cache_complete") is not True
        or not isinstance(item.get("failures"), int)
        or item["failures"] < 0
        or item.get("retries") != item["failures"]
        for item in embedding_tasks
    ):
        raise ContractError("A pilot embedding cache did not pass all completion checks.")

    tasks = payload.get("tasks", {})
    if not isinstance(tasks, dict) or set(tasks) != set(PILOT_TASKS):
        raise ContractError("Pilot modeling results do not cover exactly the three tasks.")
    expected_scopes = {"complete_history", "c2_perturbable", *(f"c3_seed_{seed}" for seed in PILOT_SEEDS)}
    expected_contrasts = {
        "B4_training_prevalence",
        "B3_governed_code_count",
        "B1_latest_window_only",
        "B2_latest_code_occurrence",
        "C0_order_only",
        "C2_reversed_earlier_gaps",
    }
    for task_report in tasks.values():
        scopes = task_report.get("scopes", {})
        if set(scopes) != expected_scopes:
            raise ContractError("A pilot task lacks one or more prespecified modeling scopes.")
        expected_representations = {
            "complete_history": {
                "B1_latest_window_only",
                "B2_latest_code_occurrence",
                "C0_order_only",
                "C1_correct_compact_time",
            },
            "c2_perturbable": {
                "C0_order_only",
                "C1_correct_compact_time",
                "C2_reversed_earlier_gaps",
            },
            **{
                f"c3_seed_{seed}": {
                    "C0_order_only",
                    "C1_correct_compact_time",
                    f"C3_shuffled_earlier_time_labels:seed={seed}",
                }
                for seed in PILOT_SEEDS
            },
        }
        for scope_name, scope_report in scopes.items():
            if set(scope_report.get("representations", {})) != expected_representations[scope_name]:
                raise ContractError("A pilot scope lacks its prespecified representations.")
            if min(scope_report.get("train_instances", 0), scope_report.get("validation_instances", 0)) < 1:
                raise ContractError("A pilot scope contains no aligned train/validation instances.")
            for representation_report in scope_report["representations"].values():
                _require_metrics(representation_report.get("metrics"), "Pilot representation")
                _require_calibration(representation_report.get("calibration"), "Pilot representation")
        b4 = task_report.get("B4_training_prevalence", {})
        _require_metrics(b4.get("metrics"), "B4")
        _require_calibration(b4.get("calibration"), "B4")
        b3 = task_report.get("B3_governed_code_count", {})
        _require_metrics(b3.get("metrics", {}).get("model"), "B3")
        _require_calibration(b3.get("calibration"), "B3")
        paired = task_report.get("paired_contrasts", [])
        if {item.get("second") for item in paired} != expected_contrasts:
            raise ContractError("A pilot task lacks one or more central paired contrasts.")
        c3 = task_report.get("c3_seed_contrasts", [])
        if {item.get("second") for item in c3} != {
            f"C3_shuffled_earlier_time_labels:seed={seed}" for seed in PILOT_SEEDS
        }:
            raise ContractError("A pilot task lacks one or more C3 seed contrasts.")
        for contrast in [*paired, *c3]:
            interval = contrast.get("ci_95")
            if (
                contrast.get("first") != "C1_correct_compact_time"
                or not isinstance(contrast.get("estimate"), int | float)
                or not isinstance(interval, list)
                or len(interval) != 2
            ):
                raise ContractError("A pilot paired contrast lacks its estimate or interval.")
        c3_summary = task_report.get("c3_seed_summary", {})
        if c3_summary.get("seeds") != list(PILOT_SEEDS):
            raise ContractError("A pilot task C3 variability summary has seed drift.")


def load_pilot_report(path: Path, repository_root: Path) -> dict[str, Any]:
    resolved = require_aggregate_output(path, repository_root)
    expected = Path("reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json")
    if resolved != (repository_root / expected).resolve():
        raise ValueError("Reviewer accepts only the configured complete-pilot aggregate report.")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pilot aggregate report must be a JSON object.")
    assert_privacy_safe_report(resolved.read_text(encoding="utf-8"))
    validate_pilot_report(payload)
    return payload


def validate_private_postconditions(payload: dict[str, Any], repository_root: Path) -> None:
    """On DGX only, hash the exact private prediction artifact without opening or printing rows."""
    predictions = (
        repository_root
        / "outputs/private/pilot/three_task_m1_engineering_pilot_v1/fit/validation_predictions.parquet"
    )
    if not predictions.is_file() or file_sha256(predictions) != payload["private_predictions_sha256"]:
        raise ContractError("Pilot private prediction artifact is missing or hash-mismatched.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = load_pilot_report(args.report, root)
    validate_private_postconditions(report, root)
    print(f"Complete engineering pilot: {report.get('status', 'UNKNOWN')}")
    print(f"Protocol decision: {report.get('protocol_decision', 'UNKNOWN')}")
    print(f"Tasks: {', '.join(report.get('scientific_scope', {}).get('tasks', []))}")
    print(f"Content matching: {'PASS' if report.get('content_matching_passed') else 'FAIL'}")
    print("Test predictions/performance used: no")
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
