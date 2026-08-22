"""Fail-closed validation for recovery of completed Phase 1 alignment artifacts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import nbformat

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.integrity import require_aggregate_output, require_private_output


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading an artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"Invalid UTC timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"UTC timestamp lacks a timezone: {value}")
    return parsed.astimezone(dt.UTC)


def inspect_executed_notebook(path: Path) -> dict[str, int]:
    """Require at least one code cell, with every code cell executed and error-free."""
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise ValueError("Executed notebook contains no code cells.")
    incomplete = [index for index, cell in enumerate(code_cells) if cell.execution_count is None]
    errors = [
        index
        for index, cell in enumerate(code_cells)
        if any(output.get("output_type") == "error" for output in cell.get("outputs", []))
    ]
    if incomplete or errors:
        raise RuntimeError(
            f"Executed notebook validation failed; incomplete_code_cells={incomplete}, error_cells={errors}."
        )
    return {
        "code_cell_count": len(code_cells),
        "completed_code_cell_count": len(code_cells),
        "error_output_count": 0,
    }


def validate_executed_notebook(path: Path) -> None:
    """Backward-compatible validator used by the standalone notebook checker."""
    inspect_executed_notebook(path)


def _validate_artifact(
    path: Path,
    *,
    repository_root: Path,
    kind: str,
    source_started_at: dt.datetime,
    source_ended_at: dt.datetime,
    expected_size: int,
    expected_mtime: dt.datetime,
) -> dict[str, Any]:
    if kind == "aggregate_report":
        resolved = require_aggregate_output(path, repository_root)
    elif kind == "executed_notebook":
        resolved = require_private_output(path, repository_root)
    else:  # pragma: no cover - internal misuse guard
        raise ValueError(f"Unsupported artifact kind: {kind}")
    if not resolved.is_file():
        raise ContractError(f"Required recovery artifact is missing: {resolved.name}")
    stat = resolved.stat()
    if stat.st_size <= 0:
        raise ContractError(f"Required recovery artifact is empty: {resolved.name}")
    if stat.st_size != expected_size:
        raise ContractError(
            f"Recovery artifact size does not match job evidence: {resolved.name}; "
            f"expected={expected_size}, actual={stat.st_size}."
        )
    observed_mtime = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.UTC).replace(microsecond=0)
    if observed_mtime != expected_mtime.replace(microsecond=0):
        raise ContractError(
            f"Recovery artifact mtime does not match job evidence: {resolved.name}; "
            f"expected={expected_mtime.isoformat()}, actual={observed_mtime.isoformat()}."
        )
    if not source_started_at <= observed_mtime <= source_ended_at:
        raise ContractError(f"Recovery artifact is stale or outside the source job interval: {resolved.name}")
    return {
        "path": str(resolved.relative_to(repository_root.resolve())),
        "size_bytes": stat.st_size,
        "mtime_utc": observed_mtime.isoformat().replace("+00:00", "Z"),
        "sha256": sha256_file(resolved),
    }


def validate_recovery_artifacts(
    *,
    repository_root: Path,
    report_path: Path,
    notebook_path: Path,
    source_failed_sentinel: Path,
    source_job_id: str,
    source_git_commit: str,
    recovery_git_commit: str,
    source_started_at: str,
    source_ended_at: str,
    expected_report_size: int,
    expected_notebook_size: int,
    expected_report_mtime: str,
    expected_notebook_mtime: str,
) -> dict[str, Any]:
    """Validate source identity, freshness, semantics, execution, and immutable failure evidence."""
    root = repository_root.resolve()
    failed = require_private_output(source_failed_sentinel, root)
    if not failed.is_file() or failed.stat().st_size <= 0:
        raise ContractError("The original source FAILED.json is missing or empty.")
    failed_bytes_before = failed.read_bytes()
    failed_sha256 = hashlib.sha256(failed_bytes_before).hexdigest()
    try:
        failed_payload = json.loads(failed_bytes_before)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("The original source FAILED.json is not valid JSON.") from exc
    expected_source = {
        "state": "FAILED",
        "job_id": str(source_job_id),
        "git_commit": source_git_commit,
        "started_at": source_started_at,
        "ended_at": source_ended_at,
    }
    mismatches = {
        key: {"expected": expected, "actual": failed_payload.get(key)}
        for key, expected in expected_source.items()
        if failed_payload.get(key) != expected
    }
    if mismatches:
        raise ContractError(f"The original failed sentinel does not match job evidence: {mismatches}")
    if not isinstance(failed_payload.get("exit_status"), int) or failed_payload["exit_status"] == 0:
        raise ContractError("The original source sentinel does not record a nonzero failure exit status.")

    started = _parse_utc(source_started_at)
    ended = _parse_utc(source_ended_at)
    if ended < started:
        raise ContractError("The source job end precedes its start.")
    report = _validate_artifact(
        report_path,
        repository_root=root,
        kind="aggregate_report",
        source_started_at=started,
        source_ended_at=ended,
        expected_size=expected_report_size,
        expected_mtime=_parse_utc(expected_report_mtime),
    )
    try:
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("The aggregate protocol-alignment report is not valid JSON.") from exc
    if not isinstance(report_payload, dict) or report_payload.get("status") != "PASS":
        raise ContractError("The aggregate protocol-alignment report does not record status PASS.")

    notebook = _validate_artifact(
        notebook_path,
        repository_root=root,
        kind="executed_notebook",
        source_started_at=started,
        source_ended_at=ended,
        expected_size=expected_notebook_size,
        expected_mtime=_parse_utc(expected_notebook_mtime),
    )
    notebook_execution = inspect_executed_notebook(notebook_path)
    if failed.read_bytes() != failed_bytes_before:
        raise ContractError("The original source FAILED.json changed during recovery validation.")

    return {
        "status": "PASS",
        "source_alignment_job_id": str(source_job_id),
        "source_git_commit": source_git_commit,
        "recovery_validator_git_commit": recovery_git_commit,
        "source_failed_sentinel": {
            "path": str(failed.relative_to(root)),
            "sha256": failed_sha256,
        },
        "artifacts": {
            "aggregate_report": report,
            "executed_notebook": {**notebook, **notebook_execution},
        },
        "postconditions": {
            "source_failed_sentinel_unchanged": True,
            "artifact_identity_matches_job_407_evidence": True,
            "aggregate_report_status_pass": True,
            "all_code_cells_completed": True,
            "no_notebook_error_outputs": True,
        },
    }


def write_validation_report(path: Path, payload: dict[str, Any], repository_root: Path) -> Path:
    """Atomically write a private, job-specific recovery validation record."""
    destination = require_private_output(path, repository_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()
    return destination
