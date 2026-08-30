"""Evidence-backed hard gates for Prompt 3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import FINAL_TASKS, FinalM1Authorization, load_final_m1_authorization
from temporal_ehr.pilot.cache import file_sha256


def _json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"Missing Prompt 3 hard-gate evidence: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Prompt 3 hard-gate evidence is not a JSON object: {path}")
    return payload


def _require_recovery_complete(payload: dict[str, Any], job_id: str) -> str:
    matching = [item for item in payload.get("jobs", []) if str(item.get("job_id")) == job_id]
    if len(matching) != 1:
        raise ContractError(f"Recovery handoff does not contain exactly one job {job_id} record.")
    complete = [
        item
        for item in matching[0].get("sentinels", [])
        if item.get("sentinel") == "COMPLETE.json"
        and item.get("state") == "COMPLETE"
        and item.get("exit_status") == 0
    ]
    if len(complete) != 1:
        raise ContractError(f"Recovery job {job_id} lacks one valid COMPLETE/exit-0 sentinel.")
    digest = complete[0].get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ContractError("Recovery COMPLETE evidence lacks a SHA-256 hash.")
    return digest


def _require_pilot_complete(payload: dict[str, Any], spec: FinalM1Authorization, root: Path) -> str:
    preconditions = spec.raw["preconditions"]
    expected_hash = str(preconditions["pilot_report_sha256"])
    reports = [
        item
        for item in payload.get("aggregate_reports", [])
        if item.get("path")
        == "reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"
    ]
    if len(reports) != 1:
        raise ContractError("Pilot handoff does not contain exactly one complete-pilot aggregate record.")
    report = reports[0]
    if (
        report.get("status") != "PASS"
        or report.get("protocol_decision") != preconditions["pilot_protocol_decision"]
        or report.get("sha256") != expected_hash
        or report.get("test_predictions_or_performance_used") is not False
        or report.get("final_panel_used_for_pilot") is not False
    ):
        raise ContractError("Complete-pilot hard-gate evidence failed or used test/final-panel information.")
    source_commit = payload.get("git", {}).get("head_commit")
    if source_commit != preconditions["pilot_source_commit"]:
        raise ContractError("Pilot handoff source commit does not match the Prompt 3 precondition.")
    local_report = root / str(report["path"])
    if not local_report.is_file() or file_sha256(local_report) != expected_hash:
        raise ContractError("Local complete-pilot aggregate is missing or hash-mismatched.")
    local = _json_object(local_report)
    if (
        local.get("status") != "PASS"
        or local.get("protocol_decision") != preconditions["pilot_protocol_decision"]
        or local.get("test_predictions_or_performance_used") is not False
        or local.get("final_panel_used_for_pilot") is not False
    ):
        raise ContractError("Local complete-pilot aggregate failed semantic hard gates.")
    return expected_hash


def require_prompt3_hard_gates(
    repository_root: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Require recovered Phase 2, complete pilot, prospective panel lock, and no test use."""
    root = repository_root.resolve()
    config = config_path or root / "configs/experiments/final_four_task_m1.yaml"
    spec = load_final_m1_authorization(config, repository_root=root)
    recovery = _json_object(root / "reports/generated/handoff/project_handoff.json")
    pilot = _json_object(root / "reports/generated/handoff/pilot_handoff.json")
    recovery_hash = _require_recovery_complete(
        recovery,
        str(spec.raw["preconditions"]["recovery_job_id"]),
    )
    pilot_hash = _require_pilot_complete(pilot, spec, root)
    if spec.tasks != FINAL_TASKS:
        raise ContractError("Final panel drifted after authorization.")
    return {
        "status": "PASS",
        "phase2_recovery_complete": True,
        "phase2_recovery_complete_sentinel_sha256": recovery_hash,
        "complete_pilot_complete": True,
        "complete_pilot_report_sha256": pilot_hash,
        "locked_tasks": list(spec.tasks),
        "panel_frozen_before_pilot_performance": True,
        "test_predictions_or_performance_used": False,
    }
