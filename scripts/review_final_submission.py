#!/usr/bin/env python3
"""Semantic, hash, and privacy validation for the one-time M1 submission workflow."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import FINAL_TASKS, load_m1_submission_authorization
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

FORBIDDEN_AGGREGATE_KEYS = {
    "subject_id",
    "patient_id",
    "instance_key",
    "prediction_time",
    "text",
    "embedding",
    "embeddings",
    "target",
    "probability",
}


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"Missing nonempty submission artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Submission artifact is not a JSON object: {path}")
    return payload


def _reject_row_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_AGGREGATE_KEYS:
                raise ContractError(f"Submission aggregate contains prohibited row field: {key}")
            _reject_row_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_row_fields(item)


def _metrics(value: Any) -> None:
    expected = {"auroc", "auprc", "brier_score", "log_loss"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError("Submission result lacks the exact metric set.")
    if any(not isinstance(item, int | float) or not math.isfinite(float(item)) for item in value.values()):
        raise ContractError("Submission result contains non-finite metrics.")


def validate_frozen(root: Path, config: Path) -> tuple[Any, dict[str, Any]]:
    spec = load_m1_submission_authorization(config, repository_root=root)
    manifest = _json(spec.aggregate_root / "FROZEN_EXPERIMENT_MANIFEST.json")
    digest = manifest.pop("frozen_manifest_sha256", None)
    if (
        digest != stable_hash(manifest)
        or manifest.get("status") != "PASS"
        or manifest.get("protocol_frozen") is not True
        or manifest.get("one_time_held_out_run_authorized") is not True
        or manifest.get("test_split_accessed") is not False
        or manifest.get("approach_scope") != spec.raw["approach_scope"]
        or manifest.get("selected_hyperparameters") is None
    ):
        raise ContractError("Frozen submission manifest is invalid.")
    manifest["frozen_manifest_sha256"] = digest
    return spec, manifest


def validate_test_cohort(spec: Any, manifest: dict[str, Any]) -> None:
    started = _json(spec.private_root / "test_access" / "STARTED.json")
    access_complete = _json(spec.private_root / "test_access" / "COMPLETE.json")
    cohort_complete = _json(spec.test_cohort_path.parent / "COMPLETE.json")
    if (
        started.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
        or access_complete.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
        or cohort_complete.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
        or not spec.test_cohort_path.is_file()
        or cohort_complete.get("cohort_sha256") != file_sha256(spec.test_cohort_path)
        or access_complete.get("cohort_sha256") != cohort_complete.get("cohort_sha256")
        or access_complete.get("performance_inspected_during_cohort_build") is not False
    ):
        raise ContractError("Held-out access or cohort evidence failed.")
    table = pq.read_table(spec.test_cohort_path, columns=["subject_id", "task", "split"])
    rows = table.to_pylist()
    if (
        {row["task"] for row in rows} != set(FINAL_TASKS)
        or {row["split"] for row in rows} != {"test"}
        or len({(row["subject_id"], row["task"]) for row in rows}) != len(rows)
    ):
        raise ContractError("Held-out cohort panel, split, or patient-balance contract failed.")


def validate_prepared(spec: Any, manifest: dict[str, Any]) -> None:
    for task in FINAL_TASKS:
        task_root = spec.private_root / "prepared" / task
        complete = _json(task_root / "COMPLETE.json")
        paths = {
            "patient_window_index_sha256": task_root / "patient_window_index.parquet",
            "patient_window_metadata_sha256": task_root / "patient_window_metadata.parquet",
            "texts_sha256": task_root / "texts.parquet",
        }
        if (
            complete.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
            or any(not path.is_file() or complete.get(field) != file_sha256(path) for field, path in paths.items())
            or complete.get("private_summary", {}).get("test_split_used") is not True
            or complete.get("private_summary", {}).get("clinical_event_arrays_rearranged_by_c2_or_c3") is not False
        ):
            raise ContractError(f"Held-out preparation failed for {task}.")


def validate_embeddings(spec: Any, manifest: dict[str, Any]) -> None:
    for task in FINAL_TASKS:
        path = spec.aggregate_root / "embedding" / f"{task}.json"
        payload = _json(path)
        _reject_row_fields(payload)
        assert_privacy_safe_report(path.read_text(encoding="utf-8"))
        if (
            payload.get("status") != "PASS"
            or payload.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
            or payload.get("task") != task
            or payload.get("condition_count") != 8
            or payload.get("full_dimension") is not True
            or payload.get("l2_normalized_float32") is not True
            or payload.get("test_split_accessed") is not True
            or payload.get("test_performance_reported") is not False
        ):
            raise ContractError(f"Held-out embedding aggregate failed for {task}.")


def validate_results(spec: Any, manifest: dict[str, Any]) -> None:
    path = spec.aggregate_root / "m1_held_out_results.json"
    payload = _json(path)
    _reject_row_fields(payload)
    assert_privacy_safe_report(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("protocol_decision")
        != "M1_PRIMARY_ONE_TIME_HELD_OUT_COMPLETE_NO_DESIGN_CHANGES_PERMITTED"
        or payload.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
        or payload.get("scope") != spec.raw["approach_scope"]
        or payload.get("test_evaluated_once") is not True
        or payload.get("protocol_modified_after_test_access") is not False
        or payload.get("test_performance_written_to_job_log") is not False
        or set(payload.get("tasks", {})) != set(FINAL_TASKS)
    ):
        raise ContractError("Held-out result identity or one-time test contract failed.")
    for task in FINAL_TASKS:
        task_report = payload["tasks"][task]
        _metrics(
            task_report["primary_scopes"]["complete_cohort"]["representations"][
                "C1_correct_compact_time"
            ]["metrics"]
        )
    _metrics(payload["equal_weight_panel_macro"]["C1_complete_cohort"]["metrics"])
    bootstrap = payload.get("shared_patient_cluster_bootstrap", {})
    if (
        bootstrap.get("requested_replicates") != 2000
        or bootstrap.get("identical_resamples_across_estimands") is not True
        or bootstrap.get("co_primary_correction") != "Holm"
        or bootstrap.get("per_task_secondary_correction") != "Benjamini-Hochberg"
        or set(bootstrap.get("task_contrasts", {})) != set(FINAL_TASKS)
    ):
        raise ContractError("Held-out shared-bootstrap or multiplicity contract failed.")
    m3 = payload.get("m3_secondary_validation_only", {})
    if (
        m3.get("held_out_accessed") is not False
        or m3.get("report_sha256") != spec.m3_report_sha256
        or set(m3.get("task_seed_aggregation_and_variability", {})) != set(FINAL_TASKS)
    ):
        raise ContractError("M3 validation-only aggregation or test boundary failed.")
    predictions = spec.private_root / "fit" / "test_predictions.parquet"
    if (
        not predictions.is_file()
        or payload.get("private_predictions_sha256") != file_sha256(predictions)
        or payload.get("private_prediction_rows") != pq.read_metadata(predictions).num_rows
    ):
        raise ContractError("Private held-out predictions fail hash or row-count validation.")


def validate_publication(root: Path, spec: Any, manifest: dict[str, Any]) -> None:
    path = spec.aggregate_root / "publication" / "PUBLICATION_COMPLETE.json"
    payload = _json(path)
    _reject_row_fields(payload)
    if (
        payload.get("status") != "PASS"
        or payload.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
        or payload.get("patient_level_artifacts_included") is not False
        or len(payload.get("artifacts", [])) < 10
    ):
        raise ContractError("Publication completion manifest failed.")
    for artifact in payload["artifacts"]:
        artifact_path = root / artifact["path"]
        if (
            not artifact_path.is_file()
            or artifact.get("sha256") != file_sha256(artifact_path)
            or artifact.get("bytes") != artifact_path.stat().st_size
        ):
            raise ContractError("Publication artifact hash or size failed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/final_m1_submission.yaml"))
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--require-test-cohort", action="store_true")
    parser.add_argument("--require-prepared", action="store_true")
    parser.add_argument("--require-embeddings", action="store_true")
    parser.add_argument("--require-results", action="store_true")
    parser.add_argument("--require-publication", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = args.config if args.config.is_absolute() else root / args.config
    spec, manifest = validate_frozen(root, config)
    if args.require_test_cohort:
        validate_test_cohort(spec, manifest)
    if args.require_prepared:
        validate_test_cohort(spec, manifest)
        validate_prepared(spec, manifest)
    if args.require_embeddings:
        validate_test_cohort(spec, manifest)
        validate_prepared(spec, manifest)
        validate_embeddings(spec, manifest)
    if args.require_results:
        validate_test_cohort(spec, manifest)
        validate_prepared(spec, manifest)
        validate_embeddings(spec, manifest)
        validate_results(spec, manifest)
    if args.require_publication:
        validate_test_cohort(spec, manifest)
        validate_prepared(spec, manifest)
        validate_embeddings(spec, manifest)
        validate_results(spec, manifest)
        validate_publication(root, spec, manifest)
    print("PASS: frozen M1 submission artifacts satisfy requested semantic, hash, and privacy gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
