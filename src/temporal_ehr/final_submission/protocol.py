"""Fail-closed protocol freeze and one-time held-out access audit."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import (
    M1SubmissionAuthorization,
    load_m1_submission_authorization,
)
from temporal_ehr.final_m1.report import _primary_scopes
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.serialization.integrity import stable_hash

FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"Missing nonempty {description}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"{description} must be a JSON object.")
    return payload


def _selected_hyperparameters(
    report: dict[str, Any],
    spec: M1SubmissionAuthorization,
) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    scopes = _primary_scopes(spec.validation)
    for task in spec.tasks:
        task_report = report.get("tasks", {}).get(task, {})
        task_scopes = task_report.get("primary_scopes", {})
        scope_values: dict[str, dict[str, float]] = {}
        for scope, labels in scopes.items():
            representation_values: dict[str, float] = {}
            for label in labels:
                value = (
                    task_scopes.get(scope, {})
                    .get("representations", {})
                    .get(label, {})
                    .get("selected_C")
                )
                if not isinstance(value, int | float) or float(value) not in (0.01, 0.1, 1.0, 10.0):
                    raise ContractError(f"Validation report lacks a frozen selected C for {task}/{scope}/{label}.")
                representation_values[label] = float(value)
            scope_values[scope] = representation_values
        b3_c = task_report.get("B3_code_count", {}).get("selected_C")
        if not isinstance(b3_c, int | float) or float(b3_c) not in (0.01, 0.1, 1.0, 10.0):
            raise ContractError(f"Validation report lacks a frozen B3 selected C for {task}.")
        selected[task] = {"scopes": scope_values, "B3_code_count": float(b3_c)}
    return selected


def require_submission_hard_gates(
    repository_root: Path,
    config_path: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Validate every pre-test input without opening held-out labels or histories."""
    root = repository_root.resolve()
    if FULL_COMMIT.fullmatch(source_git_commit) is None:
        raise ContractError("Submission freeze requires a full source Git commit.")
    spec = load_m1_submission_authorization(config_path, repository_root=root)
    m1 = _json_object(spec.validation_report_path, "final M1 validation report")
    m3 = _json_object(spec.m3_report_path, "M3 validation-only report")
    if file_sha256(spec.validation_report_path) != spec.validation_report_sha256:
        raise ContractError("Final M1 validation report hash drifted from authorization.")
    if file_sha256(spec.m3_report_path) != spec.m3_report_sha256:
        raise ContractError("M3 validation report hash drifted from authorization.")
    if (
        m1.get("status") != "PASS"
        or m1.get("protocol_decision") != "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED"
        or m1.get("test_predictions_or_performance_used") is not False
        or m1.get("test_results_reported") is not False
        or m1.get("scientific_scope", {}).get("approach") != "M1 direct serialization only"
    ):
        raise ContractError("Final M1 validation evidence is not PASS/test-locked M1 evidence.")
    if (
        m3.get("status") != "PASS"
        or m3.get("protocol_decision") != "M3_REFERENCE_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED"
        or m3.get("test_predictions_or_performance_used") is not False
        or m3.get("test_results_reported") is not False
        or m3.get("tasks") is None
    ):
        raise ContractError("M3 evidence is not PASS, validation-only, and test-locked.")
    selected = _selected_hyperparameters(m1, spec)
    access_root = spec.private_root / "test_access"
    if (access_root / "STARTED.json").exists() or (access_root / "COMPLETE.json").exists():
        raise ContractError("Held-out access already started; refusing to create another freeze.")
    if (spec.aggregate_root / "FROZEN_EXPERIMENT_MANIFEST.json").exists():
        raise ContractError("A frozen submission manifest already exists.")
    return {
        "status": "PASS",
        "source_git_commit": source_git_commit,
        "tasks": list(spec.tasks),
        "m1_validation_sha256": spec.validation_report_sha256,
        "m3_validation_sha256": spec.m3_report_sha256,
        "selected_hyperparameter_hash": stable_hash(selected),
        "test_labels_or_performance_read": False,
        "scope": spec.raw["approach_scope"],
    }


def freeze_submission_manifest(
    repository_root: Path,
    config_path: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Write the sole immutable experiment manifest before any held-out access."""
    root = repository_root.resolve()
    gates = require_submission_hard_gates(root, config_path, source_git_commit)
    spec = load_m1_submission_authorization(config_path, repository_root=root)
    m1 = _json_object(spec.validation_report_path, "final M1 validation report")
    selected = _selected_hyperparameters(m1, spec)
    body = {
        "version": 1,
        "status": "PASS",
        "protocol_decision": "M1_PRIMARY_PROTOCOL_FROZEN_ONE_TIME_TEST_AUTHORIZED",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "submission_config_sha256": file_sha256(config_path),
        "submission_config_hash": stable_hash(spec.raw),
        "base_validation_config_hash": stable_hash(spec.validation.raw),
        "tasks": list(spec.tasks),
        "split_roles": spec.raw["splits"],
        "approach_scope": spec.raw["approach_scope"],
        "fitting": spec.raw["fitting"],
        "statistics": spec.raw["statistics"],
        "representations": spec.validation.raw["representations"],
        "token_budget": spec.validation.token_budget,
        "primary_model": m1["primary_model_identity"],
        "selected_hyperparameters": selected,
        "validation_evidence": spec.raw["validation_evidence"],
        "hard_gates": gates,
        "m2_status": "INCOMPLETE_FUTURE_WORK_NOT_A_TEST_BLOCKER",
        "m3_status": "COMPLETE_SECONDARY_VALIDATION_ONLY_NO_HELD_OUT_ACCESS",
        "test_split_accessed": False,
        "protocol_frozen": True,
        "one_time_held_out_run_authorized": True,
    }
    manifest = {**body, "frozen_manifest_sha256": stable_hash(body)}
    destination = spec.aggregate_root / "FROZEN_EXPERIMENT_MANIFEST.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ContractError("Refusing to overwrite the frozen experiment manifest.")
    atomic_json(destination, manifest)
    return manifest


def load_frozen_manifest(
    repository_root: Path,
    config_path: Path,
    source_git_commit: str,
) -> tuple[M1SubmissionAuthorization, dict[str, Any]]:
    """Load and validate the exact frozen manifest used by every held-out stage."""
    root = repository_root.resolve()
    spec = load_m1_submission_authorization(config_path, repository_root=root)
    path = spec.aggregate_root / "FROZEN_EXPERIMENT_MANIFEST.json"
    manifest = _json_object(path, "frozen experiment manifest")
    digest = manifest.get("frozen_manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "frozen_manifest_sha256"}
    if (
        digest != stable_hash(body)
        or manifest.get("status") != "PASS"
        or manifest.get("protocol_frozen") is not True
        or manifest.get("one_time_held_out_run_authorized") is not True
        or manifest.get("test_split_accessed") is not False
        or manifest.get("experiment_id") != spec.experiment_id
        or manifest.get("source_git_commit") != source_git_commit
        or manifest.get("submission_config_hash") != stable_hash(spec.raw)
        or manifest.get("base_validation_config_hash") != stable_hash(spec.validation.raw)
        or manifest.get("validation_evidence") != spec.raw["validation_evidence"]
    ):
        raise ContractError("Frozen experiment manifest failed identity or hash validation.")
    _selected_hyperparameters(
        _json_object(spec.validation_report_path, "final M1 validation report"), spec
    )
    return spec, manifest


def begin_test_access(
    spec: M1SubmissionAuthorization,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Create or validate the private one-time access record before reading test labels."""
    path = spec.private_root / "test_access" / "STARTED.json"
    expected = {
        "state": "STARTED",
        "experiment_id": spec.experiment_id,
        "source_git_commit": manifest["source_git_commit"],
        "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
        "access_scope": "official held-out labels and strictly pre-prediction histories",
        "design_changes_after_access_prohibited": True,
    }
    if path.is_file():
        existing = _json_object(path, "test-access STARTED audit")
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ContractError("Existing test-access audit belongs to a different manifest.")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {**expected, "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
    return _json_object(path, "test-access STARTED audit")


def complete_test_access(
    spec: M1SubmissionAuthorization,
    manifest: dict[str, Any],
    *,
    cohort_sha256: str,
) -> dict[str, Any]:
    """Record completion of deterministic held-out cohort construction without metrics."""
    started = begin_test_access(spec, manifest)
    path = spec.private_root / "test_access" / "COMPLETE.json"
    payload = {
        **started,
        "state": "COMPLETE",
        "ended_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cohort_sha256": cohort_sha256,
        "performance_inspected_during_cohort_build": False,
    }
    if path.is_file():
        existing = _json_object(path, "test-access COMPLETE audit")
        for key in (
            "experiment_id",
            "source_git_commit",
            "frozen_manifest_sha256",
            "cohort_sha256",
        ):
            if existing.get(key) != payload.get(key):
                raise ContractError("Existing completed test-access audit drifted.")
        return existing
    atomic_json(path, payload)
    return payload
