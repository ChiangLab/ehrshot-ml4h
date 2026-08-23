#!/usr/bin/env python3
"""Build a concise, privacy-safe project handoff from strictly allowlisted metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.serialization.integrity import require_aggregate_output

AGGREGATE_REPORTS = (
    Path("reports/generated/phase1_protocol_review/protocol_alignment.json"),
    Path("reports/generated/serialization_qc/serialization_qc.json"),
    Path("reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"),
)
HASHED_PROTOCOL_FILES = (
    Path("configs/tasks.yaml"),
    Path("configs/experiments/pilot_three_task.yaml"),
    Path("configs/experiments/proposed_four_task.yaml"),
    Path("configs/experiments/locked_panel.yaml"),
    Path("configs/experiments/complete_three_task_pilot.yaml"),
    Path("docs/CURRENT_SCIENTIFIC_PROTOCOL.md"),
    Path("docs/PROTOCOL_DEVIATIONS.md"),
)
SENTINEL_NAMES = ("RUNNING.json", "COMPLETE.json", "FAILED.json")
FORBIDDEN_OUTPUT_KEYS = {
    "subject_id",
    "patient_id",
    "prediction_time",
    "raw_history",
    "raw_histories",
    "embedding",
    "embeddings",
    "label",
    "labels",
    "prediction",
    "predictions",
    "manifest_rows",
}
SAFE_PANEL_FIELDS = ("pilot_three_task", "proposed_four_task", "locked_panel")
SAFE_CHECKLIST_FIELDS = (
    "train_validation_only",
    "test_predictions_or_performance_used",
    "exact_duplicates_removed_before_serialization",
    "prediction_relative_windows_configured",
    "three_c3_seeds_checked",
    "content_matching_passed",
    "private_integrity_manifests_written",
    "experiment_panel_locked",
    "embedding_experiment_run",
    "phase3_go",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Expected a JSON object: {path.name}")
    return payload


def _git_metadata(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("Git HEAD is not a full SHA-1 commit identifier.")
    return {"head_commit": commit, "working_tree_dirty": bool(run("status", "--porcelain"))}


def _aggregate_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    checklist = payload.get("checklist")
    panel_status = payload.get("panel_status")
    summary: dict[str, Any] = {
        "path": str(path),
        "sha256": _sha256(path),
        "status": payload.get("status") if payload.get("status") in {"PASS", "FAIL"} else "UNKNOWN",
    }
    if path.name == "protocol_alignment.json":
        summary.update(
            {
                "protocol_decision": (
                    payload.get("protocol_decision")
                    if payload.get("protocol_decision") in {"GO", "NO_GO"}
                    else "UNKNOWN"
                ),
                "instances_reviewed": (
                    payload.get("instances_reviewed")
                    if isinstance(payload.get("instances_reviewed"), int) and payload["instances_reviewed"] >= 0
                    else "UNKNOWN"
                ),
                "panel_status": {
                    field: (
                        panel_status[field]
                        if panel_status[field] in {"PILOT", "PROPOSED", "PENDING", "LOCKED"}
                        else "UNKNOWN"
                    )
                    for field in SAFE_PANEL_FIELDS
                    if isinstance(panel_status, dict) and field in panel_status
                },
                "checklist": {
                    field: bool(checklist[field])
                    for field in SAFE_CHECKLIST_FIELDS
                    if isinstance(checklist, dict) and field in checklist
                },
            }
        )
    elif path.name == "pilot_results.json":
        scope = payload.get("scientific_scope", {})
        tasks = scope.get("tasks") if isinstance(scope, dict) else None
        summary.update(
            {
                "experiment_id": (
                    payload.get("experiment_id")
                    if payload.get("experiment_id") == "three_task_m1_engineering_pilot_v1"
                    else "UNKNOWN"
                ),
                "protocol_decision": (
                    payload.get("protocol_decision")
                    if payload.get("protocol_decision") == "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED"
                    else "UNKNOWN"
                ),
                "tasks": (
                    tasks
                    if tasks == ["lab_anemia", "new_acutemi", "guo_los"]
                    else []
                ),
                "model": (
                    scope.get("model")
                    if isinstance(scope, dict) and scope.get("model") == "Qwen/Qwen3-Embedding-0.6B"
                    else "UNKNOWN"
                ),
                "content_matching_passed": bool(payload.get("content_matching_passed")),
                "test_predictions_or_performance_used": bool(
                    payload.get("test_predictions_or_performance_used", True)
                ),
                "final_panel_used_for_pilot": bool(payload.get("final_panel_used_for_pilot", True)),
            }
        )
    return summary


def _environment_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    installed = payload.get("installed_dependency_audit", {})
    assets = payload.get("required_assets", {})
    tokenizer = payload.get("transformers_and_tokenizer", {})
    return {
        "sha256": _sha256(path),
        "pip_check_passed": bool(payload.get("pip_check", {}).get("passed")),
        "declared_dependencies_imported": bool(installed)
        and all(bool(item.get("import_success")) for item in installed.values()),
        "required_assets_passed": bool(assets) and all(bool(item.get("correct_type")) for item in assets.values()),
        "local_tokenizer_loaded": bool(tokenizer.get("local_load_success")),
    }


def _job_summary(root: Path, job_id: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ContractError("Job IDs may contain only letters, digits, underscores, and hyphens.")
    run_status_root = (root / "outputs/private/run_status").resolve()
    status_dir = (run_status_root / job_id).resolve()
    if not status_dir.is_relative_to(run_status_root):  # pragma: no cover - guarded by regex
        raise ContractError("Job status path escapes outputs/private/run_status/.")
    summaries: list[dict[str, Any]] = []
    environment_paths: set[Path] = set()
    for name in SENTINEL_NAMES:
        path = status_dir / name
        if not path.is_file():
            continue
        payload = _read_json(path)
        state = payload.get("state") if payload.get("state") in {"RUNNING", "COMPLETE", "FAILED"} else "UNKNOWN"
        run_kind = payload.get("run_kind")
        if run_kind not in {
            "phase1_alignment",
            "phase1_alignment_recovery",
            "pilot_prepare",
            "pilot_throughput",
            "pilot_embed_task",
            "pilot_fit_report",
        }:
            run_kind = "UNKNOWN"
        safe_payload: dict[str, Any] = {"state": state, "run_kind": run_kind}
        for field in ("job_id", "source_job_id"):
            value = payload.get(field)
            if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value):
                safe_payload[field] = value
        for field in ("git_commit", "source_git_commit", "recovery_git_commit"):
            value = payload.get(field)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
                safe_payload[field] = value
        for field in ("started_at", "ended_at"):
            value = payload.get(field)
            if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
                safe_payload[field] = value
        exit_status = payload.get("exit_status")
        if isinstance(exit_status, int) or exit_status is None:
            safe_payload["exit_status"] = exit_status
        source_sha = payload.get("source_failed_sentinel_sha256")
        if isinstance(source_sha, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha):
            safe_payload["source_failed_sentinel_sha256"] = source_sha
        summaries.append(
            {
                "sentinel": name,
                "sha256": _sha256(path),
                **safe_payload,
            }
        )
        environment_audit = payload.get("environment_audit")
        if isinstance(environment_audit, str):
            candidate = (root / environment_audit).resolve()
            if candidate.is_relative_to(run_status_root) and candidate.name == "environment_audit.json":
                environment_paths.add(candidate)
    if not summaries:
        raise ContractError(f"No allowlisted status sentinel found for job {job_id}.")
    environments = [_environment_summary(path) for path in sorted(environment_paths) if path.is_file()]
    return {"job_id": job_id, "sentinels": summaries, "environment_audits": environments}


def assert_handoff_payload_safe(payload: dict[str, Any]) -> None:
    """Reject forbidden row-level concepts even if a future collector adds them accidentally."""
    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.casefold() in FORBIDDEN_OUTPUT_KEYS:
                    raise ContractError(f"Handoff bundle contains prohibited field: {key}")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    assert_privacy_safe_report(json.dumps(payload, sort_keys=True))


def collect_handoff(repository_root: Path, job_ids: tuple[str, ...] = ()) -> dict[str, Any]:
    """Collect only Git metadata, allowlisted aggregates/status, audits, and file hashes."""
    root = repository_root.resolve()
    aggregate_reports = []
    for relative in AGGREGATE_REPORTS:
        path = root / relative
        if path.is_file():
            require_aggregate_output(path, root)
            aggregate_reports.append(_aggregate_summary(relative if relative.is_absolute() else path))
            aggregate_reports[-1]["path"] = str(relative)
    protocol_hashes = {
        str(relative): _sha256(root / relative) for relative in HASHED_PROTOCOL_FILES if (root / relative).is_file()
    }
    payload = {
        "schema_version": 1,
        "privacy_scope": "aggregate-and-metadata-only",
        "git": _git_metadata(root),
        "protocol_and_configuration_sha256": protocol_hashes,
        "aggregate_reports": aggregate_reports,
        "jobs": [_job_summary(root, job_id) for job_id in job_ids],
        "prohibited_content_included": False,
    }
    assert_handoff_payload_safe(payload)
    return payload


def write_handoff(path: Path, payload: dict[str, Any], repository_root: Path) -> Path:
    destination = require_aggregate_output(path, repository_root)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/handoff/project_handoff.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    payload = collect_handoff(root, tuple(args.job_id))
    destination = write_handoff(args.output, payload, root)
    print(f"Wrote privacy-safe handoff bundle: {destination.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
