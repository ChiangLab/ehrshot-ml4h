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
from temporal_ehr.final_m23.m2 import (
    M2_EMBEDDING_MAX_TOKENS,
    M2_SOURCE_COMPACTION_SCHEMA,
    M2_SUMMARY_CACHE_SCHEMA,
    NULL_LIKE_SUMMARIES,
    LocalSummaryTokenValidator,
    OnPremiseSummaryClient,
    _feature_sets,
    compact_exact_duplicate_facts,
    group_structured_facts,
)
from temporal_ehr.final_m23.protocol import M3_SEEDS, load_final_m23_authorization
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.serialization.integrity import stable_hash

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


def _validate_fidelity_v2_config_change(
    root: Path,
    old_commit: str,
    config_path: Path,
    expected_upstream_sha256: str,
) -> None:
    """Allow only the versioned M2 prompt/format gate after jobs 507/513."""
    relative = config_path.resolve().relative_to(root.resolve()).as_posix()
    previous_text = subprocess.run(
        ["git", "show", f"{old_commit}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if hashlib.sha256(previous_text.encode("utf-8")).hexdigest() != expected_upstream_sha256:
        raise ContractError("Upstream Prompt 4 fidelity-v2 configuration hash drifted.")
    previous = yaml.safe_load(previous_text)
    current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    previous_m2 = previous["m2"]
    current_m2 = current["m2"]
    prompt_version = current_m2.get("prompt_version")
    for field in ("prompt_version", "prompt"):
        previous_m2.pop(field)
        current_m2.pop(field)
    previous_m2["fidelity"].pop("maximum_format_violations", None)
    if (
        current_m2["fidelity"].pop("maximum_format_violations", None) != 0
        or prompt_version != "m2_fidelity_v2_compact_structured"
        or current != previous
    ):
        raise ContractError("Fidelity-v2 recovery changed more than its authorized M2 controls.")


def _validate_fidelity_v3_config_change(
    root: Path,
    old_commit: str,
    config_path: Path,
    expected_upstream_sha256: str,
) -> None:
    """Allow only grouped M2 prompt/output-budget controls after failed array 519."""
    relative = config_path.resolve().relative_to(root.resolve()).as_posix()
    previous_text = subprocess.run(
        ["git", "show", f"{old_commit}:{relative}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if hashlib.sha256(previous_text.encode("utf-8")).hexdigest() != expected_upstream_sha256:
        raise ContractError("Upstream Prompt 4 fidelity-v3 configuration hash drifted.")
    previous = yaml.safe_load(previous_text)
    current = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    previous_m2 = previous["m2"]
    current_m2 = current["m2"]
    previous_controls = {
        "prompt_version": previous_m2.pop("prompt_version"),
        "prompt": previous_m2.pop("prompt"),
        "max_output_tokens": previous_m2["decoding"].pop("max_output_tokens"),
    }
    current_controls = {
        "prompt_version": current_m2.pop("prompt_version"),
        "prompt": current_m2.pop("prompt"),
        "max_output_tokens": current_m2["decoding"].pop("max_output_tokens"),
    }
    if (
        previous_controls["prompt_version"] != "m2_fidelity_v2_compact_structured"
        or previous_controls["max_output_tokens"] != 2048
        or current_controls["prompt_version"] != "m2_fidelity_v3_grouped_structured"
        or current_controls["max_output_tokens"] != 4096
        or not isinstance(current_controls["prompt"], str)
        or "already been deterministically grouped" not in current_controls["prompt"]
        or current != previous
    ):
        raise ContractError("Fidelity-v3 recovery changed more than its authorized M2 controls.")


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


def _validate_failed_status(
    root: Path,
    status_id: str,
    commit: str,
    run_kind: str,
    expected_exit_status: int,
) -> dict[str, str]:
    directory = root / "outputs/private/run_status" / status_id
    running_path = directory / "RUNNING.json"
    failed_path = directory / "FAILED.json"
    running = _json_object(running_path, f"job-{status_id} RUNNING sentinel")
    failed = _json_object(failed_path, f"job-{status_id} FAILED sentinel")
    if (
        running.get("state") != "RUNNING"
        or running.get("git_commit") != commit
        or running.get("run_kind") != run_kind
        or failed.get("state") != "FAILED"
        or failed.get("git_commit") != commit
        or failed.get("run_kind") != run_kind
        or failed.get("exit_status") != expected_exit_status
        or (directory / "COMPLETE.json").exists()
    ):
        raise ContractError(f"Job {status_id} failed exact failed-attempt provenance checks.")
    return {
        "RUNNING.json": file_sha256(running_path),
        "FAILED.json": file_sha256(failed_path),
    }


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


def validate_prompt4_m2_fidelity_v2_recovery(
    *,
    repository_root: Path,
    config_path: Path,
    authorization_path: Path,
    current_source_git_commit: str,
    live_endpoint_check: bool,
) -> dict[str, Any]:
    """Gate the post-507/513 compact-summary recovery without exposing governed rows."""
    root = repository_root.resolve()
    if FULL_COMMIT.fullmatch(current_source_git_commit) is None:
        raise ContractError("M2 fidelity-v2 recovery requires a full current Git commit.")
    auth = _load_authorization(root, authorization_path)
    old_commit = str(auth.get("upstream_source_git_commit", ""))
    expected_controls = {
        "preserve_superseded_jobs_and_caches": True,
        "version_prompt_and_cache_identity": True,
        "require_finish_reason_stop": True,
        "record_token_usage": True,
        "parse_plain_clinical_codes": True,
        "reject_unstructured_expansion": True,
        "stable_identity_excludes_service_start_timestamp": True,
        "deterministic_exact_repeat_precompaction": True,
        "source_compaction_in_cache_identity": True,
        "consecutive_dense_synthetic_checks": 3,
        "rerun_m2": True,
        "rerun_m3": False,
        "test_predictions_or_performance_used": False,
    }
    if (
        auth.get("version") != 2
        or auth.get("mode") != "M2_FIDELITY_V2_RECOVERY"
        or auth.get("experiment_id") != "locked_four_task_m2_m3_v1"
        or FULL_COMMIT.fullmatch(old_commit) is None
        or current_source_git_commit == old_commit
        or auth.get("controls") != expected_controls
        or auth.get("failed_preflight")
        != {
            "source_git_commit": "f25080d69c01b840522fa83b27004a2879c8d697",
            "governed_jobs_submitted": False,
            "recovery_report_written": False,
            "synthetic_requested_occurrences": 64,
            "synthetic_observed_incorrect_occurrences": 65,
        }
    ):
        raise ContractError("Prompt 4 M2 fidelity-v2 recovery authorization drifted.")
    spec = load_final_m23_authorization(config_path, repository_root=root)
    _validate_fidelity_v2_config_change(
        root,
        old_commit,
        config_path,
        str(auth.get("upstream_experiment_config_sha256", "")),
    )

    source_jobs = auth.get("source_jobs")
    if source_jobs != {
        "superseded_m2_summary_array": 507,
        "completed_reference_index": 1,
        "canceled_task0_recovery": 513,
        "m3_preparation": {"job_id": 470, "indices": [0, 1, 2, 3]},
        "m3_seed_array": {"job_id": 471, "indices": list(range(12))},
    }:
        raise ContractError("Fidelity-v2 source-job authorization drifted.")
    completed_507_1 = _validate_complete_status(
        root,
        "507_1",
        old_commit,
        "final_m2_summary_reference",
    )
    preserved_status_hashes: dict[str, dict[str, str]] = {}
    for status_id in ("507_0", "507_1", "507_2", "507_3", "513_0"):
        directory = root / "outputs/private/run_status" / status_id
        hashes = {
            path.name: file_sha256(path)
            for path in directory.glob("*.json")
            if path.name in {"RUNNING.json", "COMPLETE.json", "FAILED.json"}
        }
        if hashes:
            preserved_status_hashes[status_id] = hashes

    m3_status_hashes = {
        f"470_{index}": _validate_complete_status(
            root, f"470_{index}", "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913", "final_m3_prepare_reference"
        )
        for index in range(4)
    }
    m3_status_hashes.update(
        {
            f"471_{index}": _validate_complete_status(
                root,
                f"471_{index}",
                "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913",
                "final_m3_task_seed_reference",
            )
            for index in range(12)
        }
    )

    prompt_hash = stable_hash({"version": spec.m2["prompt_version"], "prompt": spec.m2["prompt"]})
    legacy_rows = 0
    current_rows = 0
    for path in (spec.private_root / "m2" / "summaries").glob("*/rows/*.json"):
        payload = _json_object(path, "M2 summary cache row")
        if payload.get("prompt_hash") == prompt_hash:
            current_rows += 1
        else:
            legacy_rows += 1
    if legacy_rows < 1 or current_rows != 0:
        raise ContractError("Fidelity-v2 preflight requires legacy evidence and an unused prompt hash.")

    live_summary_hashes: list[str] = []
    model_identity_hash = None
    live_completion_tokens: list[int | None] = []
    if live_endpoint_check:
        client = OnPremiseSummaryClient(
            api_base=str(spec.m2["api_base"]), model=str(spec.m2["instruction_model"])
        )
        model_identity_hash = client.model_revision_hash()
        repeated = "\n".join(
            "- LOINC/SYNTHETIC-A | value=0.1 | unit=unit-A | source=measurement"
            for _ in range(64)
        )
        uncompressed_source = (
            "[TIME=31_60_DAYS_BEFORE_T]\n"
            f"{repeated}\n"
            "- CPT4/SYNTHETIC-B | source=procedure\n"
            "[TIME=1_30_DAYS_BEFORE_T]\n"
            "- SNOMED/SYNTHETIC-C | value=2 | unit=unit-C | source=measurement"
        )
        source = compact_exact_duplicate_facts(uncompressed_source)
        source_features = _feature_sets(source)
        if source_features["recurrence"].get("LOINC/SYNTHETIC-A") != 64:
            raise ContractError("Deterministic M2 preflight compaction lost synthetic recurrence.")
        required_concepts = {"LOINC/SYNTHETIC-A", "CPT4/SYNTHETIC-B", "SNOMED/SYNTHETIC-C"}
        for attempt in range(3):
            generation = client.summarize_response(
                system_prompt=str(spec.m2["prompt"]),
                source_text=source,
                decoding=spec.m2["decoding"],
            )
            features = _feature_sets(generation.summary)
            checks = {
                "concepts": set(features["concepts"]) == required_concepts,
                "recurrence": features["recurrence"].get("LOINC/SYNTHETIC-A") == 64,
                "time_order": features["time_label_sequence"]
                == ["[TIME=31_60_DAYS_BEFORE_T]", "[TIME=1_30_DAYS_BEFORE_T]"],
                "format": features["format_violations"] == 0,
                "numbers": {"0.1", "2"} <= set(features["numbers"]),
                "units": set(features["units"]) == {"unit-A", "unit-C"},
            }
            if not all(checks.values()):
                failed = ",".join(name for name, passed in checks.items() if not passed)
                raise ContractError(
                    f"Dense synthetic M2 summary attempt {attempt + 1} failed: {failed}."
                )
            live_summary_hashes.append(stable_hash(generation.summary))
            live_completion_tokens.append(generation.completion_tokens)

    report = {
        "status": "PASS",
        "mode": "M2_FIDELITY_V2_RECOVERY",
        "source_git_commit": current_source_git_commit,
        "superseded_m2_git_commit": old_commit,
        "m3_compute_git_commit": "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913",
        "completed_507_1_sentinel_sha256": completed_507_1,
        "preserved_superseded_status_sha256": preserved_status_hashes,
        "m3_complete_sentinel_sha256": m3_status_hashes,
        "legacy_summary_cache_rows": legacy_rows,
        "new_prompt_cache_rows_before_submission": current_rows,
        "prompt_version": spec.m2["prompt_version"],
        "source_compaction_schema": M2_SOURCE_COMPACTION_SCHEMA,
        "live_dense_synthetic_repetitions": 3 if live_endpoint_check else 0,
        "finish_reason_stop_required": True,
        "stable_served_model_identity_hash": model_identity_hash,
        "live_dense_synthetic_check_passed": live_endpoint_check,
        "live_dense_synthetic_summary_sha256": live_summary_hashes,
        "live_dense_synthetic_completion_tokens": live_completion_tokens,
        "m3_rerun_required": False,
        "test_predictions_or_performance_used": False,
        "patient_content_included": False,
    }
    assert_privacy_safe_report(json.dumps(report, sort_keys=True))
    return report


def validate_prompt4_m2_fidelity_v3_recovery(
    *,
    repository_root: Path,
    config_path: Path,
    authorization_path: Path,
    model_root: Path,
    current_source_git_commit: str,
    live_endpoint_check: bool,
) -> dict[str, Any]:
    """Gate the grouped post-519 M2 recovery while reusing canonical M3 evidence."""
    root = repository_root.resolve()
    if FULL_COMMIT.fullmatch(current_source_git_commit) is None:
        raise ContractError("M2 fidelity-v3 recovery requires a full current Git commit.")
    auth = _load_authorization(root, authorization_path)
    old_commit = str(auth.get("upstream_source_git_commit", ""))
    expected_controls = {
        "preserve_job_519_status_and_caches": True,
        "preserve_canceled_downstream_jobs": True,
        "group_by_code_source_unit_within_window": True,
        "preserve_distinct_values_first_occurrence_order": True,
        "preserve_total_occurrence_count": True,
        "version_prompt_compaction_and_cache_identity": True,
        "require_finish_reason_stop": True,
        "require_downstream_8192_token_fit": True,
        "progress_interval_rows": 25,
        "consecutive_dense_synthetic_checks": 3,
        "rerun_m2": True,
        "rerun_m3": False,
        "test_predictions_or_performance_used": False,
    }
    expected_jobs = {
        "failed_m2_summary_array": {
            "job_id": 519,
            "indices": [0, 1, 2, 3],
            "exit_status_by_index": {0: 143, 1: 143, 2: 143, 3: 1},
        },
        "canceled_m2_embedding": {"job_id": 520},
        "canceled_m2_review": {"job_id": 521},
        "m3_preparation": {"job_id": 470, "indices": [0, 1, 2, 3]},
        "m3_seed_array": {"job_id": 471, "indices": list(range(12))},
    }
    if (
        auth.get("version") != 3
        or auth.get("mode") != "M2_FIDELITY_V3_RECOVERY"
        or auth.get("experiment_id") != "locked_four_task_m2_m3_v1"
        or FULL_COMMIT.fullmatch(old_commit) is None
        or current_source_git_commit == old_commit
        or auth.get("controls") != expected_controls
        or auth.get("source_jobs") != expected_jobs
    ):
        raise ContractError("Prompt 4 M2 fidelity-v3 recovery authorization drifted.")

    spec = load_final_m23_authorization(config_path, repository_root=root)
    _validate_fidelity_v3_config_change(
        root,
        old_commit,
        config_path,
        str(auth.get("upstream_experiment_config_sha256", "")),
    )
    failed_status_hashes = {
        f"519_{index}": _validate_failed_status(
            root,
            f"519_{index}",
            old_commit,
            "final_m2_summary_reference",
            int(expected_jobs["failed_m2_summary_array"]["exit_status_by_index"][index]),
        )
        for index in range(4)
    }

    for downstream_id in ("520", "521"):
        status_root = root / "outputs/private/run_status"
        status_names = [downstream_id, *(f"{downstream_id}_{index}" for index in range(4))]
        if any((status_root / name / "COMPLETE.json").exists() for name in status_names):
            raise ContractError(f"Canceled downstream job {downstream_id} unexpectedly has COMPLETE evidence.")

    m3_status_hashes = {
        f"470_{index}": _validate_complete_status(
            root,
            f"470_{index}",
            "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913",
            "final_m3_prepare_reference",
        )
        for index in range(4)
    }
    m3_status_hashes.update(
        {
            f"471_{index}": _validate_complete_status(
                root,
                f"471_{index}",
                "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913",
                "final_m3_task_seed_reference",
            )
            for index in range(12)
        }
    )

    current_prompt_hash = stable_hash(
        {"version": spec.m2["prompt_version"], "prompt": spec.m2["prompt"]}
    )
    legacy_rows = 0
    current_rows = 0
    for path in (spec.private_root / "m2" / "summaries").glob("*/rows/*.json"):
        payload = _json_object(path, "M2 summary cache row")
        if payload.get("prompt_hash") == current_prompt_hash:
            current_rows += 1
        else:
            legacy_rows += 1
    if legacy_rows < 1 or current_rows != 0:
        raise ContractError("Fidelity-v3 preflight requires preserved legacy rows and an unused prompt hash.")

    live_summary_hashes: list[str] = []
    live_completion_tokens: list[int | None] = []
    live_embedding_tokens: list[int] = []
    model_identity_hash = None
    if live_endpoint_check:
        client = OnPremiseSummaryClient(
            api_base=str(spec.m2["api_base"]),
            model=str(spec.m2["instruction_model"]),
        )
        model_identity_hash = client.model_revision_hash()
        repeated_a = "\n".join(
            "- LOINC/SYNTHETIC-A | value=0.1 | unit=unit-A | source=measurement"
            for _ in range(64)
        )
        repeated_b = "\n".join(
            f"- LOINC/SYNTHETIC-A | value={value} | unit=unit-A | source=measurement"
            for value in ("0.2", "0.1", "0.3", "0.2")
        )
        ungrouped = (
            "[TIME=31_60_DAYS_BEFORE_T]\n"
            f"{repeated_a}\n{repeated_b}\n"
            "- CPT4/SYNTHETIC-B | text_value=present | source=procedure\n"
            "- CPT4/SYNTHETIC-B | text_value=absent | source=procedure\n"
            "[TIME=1_30_DAYS_BEFORE_T]\n"
            "- SNOMED/SYNTHETIC-C | value=2 | unit=unit-C | source=measurement"
        )
        grouped = group_structured_facts(ungrouped)
        expected_grouped = (
            "[TIME=31_60_DAYS_BEFORE_T]\n"
            "- LOINC/SYNTHETIC-A | values=0.1,0.2,0.3 | unit=unit-A | "
            "source=measurement | occurrences=68\n"
            "- CPT4/SYNTHETIC-B | text_values=[\"present\",\"absent\"] | "
            "source=procedure | occurrences=2\n"
            "[TIME=1_30_DAYS_BEFORE_T]\n"
            "- SNOMED/SYNTHETIC-C | values=2 | unit=unit-C | "
            "source=measurement | occurrences=1"
        )
        if grouped != expected_grouped:
            raise ContractError("Deterministic M2 grouped-source preflight drifted.")
        token_validator = LocalSummaryTokenValidator(
            model_root / spec.model_directory_name,
            maximum_tokens=M2_EMBEDDING_MAX_TOKENS,
        )
        for attempt in range(3):
            generation = client.summarize_response(
                system_prompt=str(spec.m2["prompt"]),
                source_text=grouped,
                decoding=spec.m2["decoding"],
            )
            if generation.summary.strip() != grouped:
                raise ContractError(
                    f"Dense grouped M2 summary attempt {attempt + 1} did not copy every field exactly."
                )
            live_summary_hashes.append(stable_hash(generation.summary))
            live_completion_tokens.append(generation.completion_tokens)
            live_embedding_tokens.append(token_validator(generation.summary))

    report = {
        "status": "PASS",
        "mode": "M2_FIDELITY_V3_RECOVERY",
        "source_git_commit": current_source_git_commit,
        "superseded_m2_git_commit": old_commit,
        "m3_compute_git_commit": "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913",
        "preserved_job_519_status_sha256": failed_status_hashes,
        "canceled_downstream_jobs": [520, 521],
        "m3_complete_sentinel_sha256": m3_status_hashes,
        "legacy_summary_cache_rows": legacy_rows,
        "new_prompt_cache_rows_before_submission": current_rows,
        "prompt_version": spec.m2["prompt_version"],
        "source_compaction_schema": M2_SOURCE_COMPACTION_SCHEMA,
        "summary_cache_schema": M2_SUMMARY_CACHE_SCHEMA,
        "maximum_output_tokens": int(spec.m2["decoding"]["max_output_tokens"]),
        "thinking_disabled": spec.m2["decoding"]["chat_template_kwargs"]["enable_thinking"]
        is False,
        "embedding_summary_max_tokens": M2_EMBEDDING_MAX_TOKENS,
        "live_dense_synthetic_repetitions": 3 if live_endpoint_check else 0,
        "live_dense_synthetic_check_passed": live_endpoint_check,
        "live_dense_synthetic_summary_sha256": live_summary_hashes,
        "live_dense_synthetic_completion_tokens": live_completion_tokens,
        "live_dense_synthetic_embedding_tokens": live_embedding_tokens,
        "stable_served_model_identity_hash": model_identity_hash,
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
