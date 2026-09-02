"""Fail-closed recovery validation for the Prompt 4 M2 null-content defect."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23.m2 import NULL_LIKE_SUMMARIES, OnPremiseSummaryClient
from temporal_ehr.final_m23.protocol import M3_SEEDS, load_final_m23_authorization
from temporal_ehr.pilot.cache import atomic_json, file_sha256

FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"Prompt 4 recovery requires a nonempty {label}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Prompt 4 recovery {label} must be a JSON object.")
    return payload


def _load_authorization(root: Path, path: Path) -> dict[str, Any]:
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to((root / "configs/recovery").resolve()):
        raise ContractError("Prompt 4 recovery authorization must remain under configs/recovery.")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("Prompt 4 recovery authorization must be a YAML mapping.")
    return payload


def _validate_only_allowed_config_change(
    root: Path,
    old_commit: str,
    config_path: Path,
    expected_upstream_sha256: str | None = None,
) -> None:
    relative = config_path.resolve().relative_to(root.resolve()).as_posix()
    try:
        previous_text = subprocess.run(
            ["git", "show", f"{old_commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ContractError("Unable to read the upstream Prompt 4 configuration from Git.") from exc
    previous = yaml.safe_load(previous_text)
    previous_sha256 = hashlib.sha256(previous_text.encode("utf-8")).hexdigest()
    if expected_upstream_sha256 is not None and previous_sha256 != expected_upstream_sha256:
        raise ContractError("Upstream Prompt 4 configuration hash drifted from recovery authorization.")
    current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kwargs = current.get("m2", {}).get("decoding", {}).pop("chat_template_kwargs", None)
    if kwargs != {"enable_thinking": False} or current != previous:
        raise ContractError("Recovery changed more than the locked M2 non-thinking decoding control.")


def _validate_complete_status(root: Path, status_id: str, commit: str, run_kind: str) -> str:
    directory = root / "outputs/private/run_status" / status_id
    payload = _json_object(directory / "COMPLETE.json", f"job-{status_id} COMPLETE sentinel")
    if (
        payload.get("state") != "COMPLETE"
        or payload.get("exit_status") != 0
        or payload.get("git_commit") != commit
        or payload.get("run_kind") != run_kind
        or (directory / "FAILED.json").exists()
    ):
        raise ContractError(f"Job {status_id} failed exact recovery provenance checks.")
    return file_sha256(directory / "COMPLETE.json")


def _audit_invalid_cache_rows(root: Path) -> dict[str, Any]:
    summary_root = root / "outputs/private/final/locked_four_task_m2_m3_v1/m2/summaries"
    counts: Counter[str] = Counter()
    for path in summary_root.glob("*/rows/*.json"):
        payload = _json_object(path, "legacy M2 summary cache row")
        summary = payload.get("summary")
        if not isinstance(summary, str):
            counts["non_text"] += 1
        elif not summary.strip() or summary.strip().casefold() in NULL_LIKE_SUMMARIES:
            counts["empty_or_null_like"] += 1
        else:
            counts["nonempty_text"] += 1
    invalid = counts["non_text"] + counts["empty_or_null_like"]
    if invalid < 1:
        raise ContractError("Recovery did not find the documented invalid M2 cache rows.")
    return {
        "rows_scanned": sum(counts.values()),
        "invalid_rows": invalid,
        "nonempty_text_rows": counts["nonempty_text"],
        "patient_content_included": False,
    }


def validate_prompt4_m2_recovery(
    *,
    repository_root: Path,
    config_path: Path,
    authorization_path: Path,
    current_source_git_commit: str,
    live_endpoint_check: bool,
) -> dict[str, Any]:
    """Validate job 502, M3 reuse, cache isolation, and the live non-thinking API contract."""
    root = repository_root.resolve()
    if FULL_COMMIT.fullmatch(current_source_git_commit) is None:
        raise ContractError("Prompt 4 recovery requires a full current Git commit.")
    auth = _load_authorization(root, authorization_path)
    old_commit = str(auth.get("upstream_source_git_commit", ""))
    expected_controls = {
        "preserve_job_502_status": True,
        "reject_null_like_summary_content": True,
        "disable_instruction_model_thinking": True,
        "isolate_recovery_cache_by_decoding_hash": True,
        "rerun_m2": True,
        "rerun_m3": False,
        "test_predictions_or_performance_used": False,
    }
    if (
        auth.get("version") != 1
        or auth.get("mode") != "M2_NULL_CONTENT_RECOVERY"
        or auth.get("experiment_id") != "locked_four_task_m2_m3_v1"
        or FULL_COMMIT.fullmatch(old_commit) is None
        or current_source_git_commit == old_commit
        or auth.get("controls") != expected_controls
        or auth.get("source_jobs")
        != {
            "incomplete_m2_summary": {"job_id": 502, "array_index": 0},
            "canceled_m2_embedding": {"job_id": 504},
            "canceled_m2_review": {"job_id": 505},
            "m3_preparation": {"job_id": 470, "indices": [0, 1, 2, 3]},
            "m3_seed_array": {"job_id": 471, "indices": list(range(12))},
        }
    ):
        raise ContractError("Prompt 4 M2 recovery authorization drifted.")
    spec = load_final_m23_authorization(config_path, repository_root=root)
    _validate_only_allowed_config_change(
        root,
        old_commit,
        config_path,
        str(auth.get("upstream_experiment_config_sha256", "")),
    )

    job_502 = root / "outputs/private/run_status/502_0"
    running = _json_object(job_502 / "RUNNING.json", "job-502_0 RUNNING sentinel")
    if (
        running.get("state") != "RUNNING"
        or running.get("git_commit") != old_commit
        or running.get("run_kind") != "final_m2_summary_reference"
        or (job_502 / "COMPLETE.json").exists()
        or (job_502 / "FAILED.json").exists()
    ):
        raise ContractError("Job 502_0 is not the preserved ended-without-final-sentinel attempt.")
    job_502_hash = file_sha256(job_502 / "RUNNING.json")

    m3_status_hashes = {
        f"470_{index}": _validate_complete_status(
            root, f"470_{index}", old_commit, "final_m3_prepare_reference"
        )
        for index in range(4)
    }
    m3_status_hashes.update(
        {
            f"471_{index}": _validate_complete_status(
                root, f"471_{index}", old_commit, "final_m3_task_seed_reference"
            )
            for index in range(12)
        }
    )
    m3_report_hashes: dict[str, str] = {}
    for task in spec.tasks:
        for seed in M3_SEEDS:
            path = spec.aggregate_root / "m3" / "tasks" / f"{task}_seed_{seed}_reference.json"
            payload = _json_object(path, "M3 task/seed report")
            if (
                payload.get("status") != "PASS"
                or payload.get("source_git_commit") != old_commit
                or payload.get("task") != task
                or payload.get("seed") != seed
                or payload.get("test_split_used") is not False
            ):
                raise ContractError("An M3 report failed recovery provenance checks.")
            m3_report_hashes[f"{task}:{seed}"] = file_sha256(path)

    cache_audit = _audit_invalid_cache_rows(root)
    live_summary_hash = None
    model_revision_hash = None
    if live_endpoint_check:
        client = OnPremiseSummaryClient(
            api_base=str(spec.m2["api_base"]), model=str(spec.m2["instruction_model"])
        )
        model_revision_hash = client.model_revision_hash()
        source = (
            "[TIME=60_90_DAYS_BEFORE_T]: [SYNTHETIC_LAB_X] value=10 unit=g/dL\n"
            "[TIME=30_60_DAYS_BEFORE_T]: [SYNTHETIC_LAB_X] value=11 unit=g/dL"
        )
        summary = client.summarize(
            system_prompt=str(spec.m2["prompt"]),
            source_text=source,
            decoding=spec.m2["decoding"],
        )
        required = ("[TIME=60_90_DAYS_BEFORE_T]", "[TIME=30_60_DAYS_BEFORE_T]", "[SYNTHETIC_LAB_X]", "10", "11", "g/dL")
        if not all(value in summary for value in required):
            raise ContractError("Live M2 recovery preflight did not preserve every synthetic field.")
        from temporal_ehr.serialization.integrity import stable_hash

        live_summary_hash = stable_hash(summary)

    report = {
        "status": "PASS",
        "mode": "M2_NULL_CONTENT_RECOVERY",
        "source_git_commit": current_source_git_commit,
        "m3_compute_git_commit": old_commit,
        "preserved_job_502_running_sentinel_sha256": job_502_hash,
        "job_502_final_sentinel_present": False,
        "legacy_cache_audit": cache_audit,
        "legacy_cache_reuse_prevented_by_new_decoding_hash": True,
        "thinking_disabled": True,
        "live_endpoint_check_passed": live_endpoint_check,
        "live_synthetic_summary_sha256": live_summary_hash,
        "instruction_model_revision_hash": model_revision_hash,
        "m3_complete_sentinel_sha256": m3_status_hashes,
        "m3_task_report_sha256": m3_report_hashes,
        "m3_rerun_required": False,
        "test_predictions_or_performance_used": False,
        "patient_content_included": False,
    }
    assert_privacy_safe_report(json.dumps(report, sort_keys=True))
    return report


def write_recovery_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise ContractError("Refusing to overwrite an existing Prompt 4 recovery report.")
    atomic_json(path, report)
