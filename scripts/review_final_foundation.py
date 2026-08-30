#!/usr/bin/env python3
"""Semantic review of the privacy-safe final-panel foundation aggregate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


def _reject_row_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_ROW_KEYS:
                raise ContractError(f"Final foundation aggregate contains prohibited row field: {key}")
            _reject_row_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_row_fields(item)


def validate_final_foundation(payload: dict[str, Any]) -> None:
    _reject_row_fields(payload)
    if payload.get("status") != "PASS" or payload.get("protocol_decision") != (
        "FINAL_FOUNDATION_COMPLETE_M1_AUTHORIZED"
    ):
        raise ContractError("Final foundation lacks its exact PASS decision.")
    if payload.get("pilot_artifacts_reused") is not False or payload.get(
        "test_predictions_or_performance_used"
    ) is not False:
        raise ContractError("Final foundation reused pilot artifacts or used test information.")
    hard = payload.get("hard_gates", {})
    if (
        hard.get("phase2_recovery_complete") is not True
        or hard.get("complete_pilot_complete") is not True
        or hard.get("locked_tasks") != list(FINAL_TASKS)
        or hard.get("panel_frozen_before_pilot_performance") is not True
        or hard.get("test_predictions_or_performance_used") is not False
    ):
        raise ContractError("Final foundation hard-gate evidence is incomplete.")
    scope = payload.get("scientific_scope", {})
    if (
        scope.get("tasks") != list(FINAL_TASKS)
        or scope.get("splits") != ["train", "validation"]
        or scope.get("token_budget") != 8192
        or scope.get("window_days_by_task")
        != {"lab_anemia": 30, "guo_readmission": 30, "new_hypertension": 90, "chexpert": 30}
        or scope.get("strict_censoring_rule") != "event_time < prediction_time"
        or len(scope.get("exact_duplicate_fields", [])) != 7
    ):
        raise ContractError("Final foundation task/window/censoring/deduplication scope drifted.")
    cohort = payload.get("cohort", {})
    groups = cohort.get("groups", []) if isinstance(cohort, dict) else []
    if (
        {(item.get("task"), item.get("split")) for item in groups}
        != {(task, split) for task in FINAL_TASKS for split in ("train", "validation")}
        or cohort.get("maximum_instances_per_patient_task") != 1
        or cohort.get("test_instances_selected") != 0
        or cohort.get("selection_uses_outcome") is not False
        or cohort.get("selection_uses_history_or_record_length") is not False
        or cohort.get("selection_uses_model_result") is not False
    ):
        raise ContractError("Final cohort selection or coverage contract failed.")
    summaries = payload.get("task_summaries", [])
    if {item.get("task") for item in summaries} != set(FINAL_TASKS):
        raise ContractError("Final foundation summaries do not cover exactly four tasks.")
    for summary in summaries:
        if (
            summary.get("source_instances") != summary.get("representations", {}).get(
                "C1_correct_compact_time"
            )
            or summary.get("metadata_mirrors_every_embedding_row") is not True
            or summary.get("content_matching_passed") is not True
            or summary.get("clinical_event_arrays_rearranged_by_c2_or_c3") is not False
            or summary.get("test_split_used") is not False
            or not isinstance(summary.get("representation_coverage"), dict)
            or not isinstance(summary.get("perturbation_strength"), dict)
        ):
            raise ContractError("One final preparation summary failed alignment/content/coverage gates.")
    audits = payload.get("outcome_and_shortcut_audits", {})
    if set(audits) != set(FINAL_TASKS):
        raise ContractError("Outcome/shortcut audits do not cover exactly four tasks.")
    if any(
        item.get("status") != "PASS"
        or item.get("official_prediction_time_used") is not True
        or item.get("all_retained_events_strictly_before_prediction_time") is not True
        or item.get("target_defining_future_events_retained") != 0
        or item.get("prior_related_predictors_before_prediction_time_allowed") is not True
        for item in audits.values()
    ):
        raise ContractError("One outcome/shortcut audit failed.")
    if (
        payload.get("count_baseline", {}).get("status") != "PASS"
        or payload.get("count_baseline", {}).get("test_split_inspected") is not False
        or payload.get("content_matching_passed") is not True
        or payload.get("label_index_only_perturbations_passed") is not True
        or payload.get("patient_window_metadata_alignment_passed") is not True
    ):
        raise ContractError("Final foundation baseline or serializer postcondition failed.")


def validate_private_postconditions(payload: dict[str, Any], root: Path) -> None:
    cohort = root / "outputs/private/final/locked_four_task_m1_v1/cohort/final_panel_cohort_v1.parquet"
    if not cohort.is_file() or file_sha256(cohort) != payload.get("cohort_sha256"):
        raise ContractError("Final cohort is missing or hash-mismatched.")
    for task in FINAL_TASKS:
        task_root = root / f"outputs/private/final/locked_four_task_m1_v1/prepared/{task}"
        complete = json.loads((task_root / "COMPLETE.json").read_text(encoding="utf-8"))
        expected = {
            "patient_window_index_sha256": task_root / "patient_window_index.parquet",
            "patient_window_metadata_sha256": task_root / "patient_window_metadata.parquet",
            "texts_sha256": task_root / "texts.parquet",
        }
        if any(not path.is_file() or complete.get(field) != file_sha256(path) for field, path in expected.items()):
            raise ContractError(f"Final private preparation hash failed for {task}.")


def load_final_foundation(path: Path, root: Path, *, private_postconditions: bool) -> dict[str, Any]:
    destination = require_aggregate_output(path, root)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert_privacy_safe_report(json.dumps(payload, sort_keys=True))
    validate_final_foundation(payload)
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
    payload = load_final_foundation(report, root, private_postconditions=args.private_postconditions)
    print(f"PASS: reviewed final foundation for {len(payload['task_summaries'])} tasks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
