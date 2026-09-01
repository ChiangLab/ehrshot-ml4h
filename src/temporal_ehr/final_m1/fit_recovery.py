"""Fail-closed provenance validation for final-M1 fitting and fit-only recovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import FINAL_TASKS, FinalM1Authorization
from temporal_ehr.final_m1.throughput import validate_foundation_reuse
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class ValidatedFitInputs:
    """Validated aggregate inputs and their privacy-safe provenance."""

    foundation: dict[str, Any]
    throughput: dict[str, Any]
    embedding_reports: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"Final fit requires a nonempty {description}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError(f"Final fit {description} must be a JSON object.")
    return payload


def _validate_status(
    repository_root: Path,
    *,
    status_id: str,
    state: str,
    source_git_commit: str,
    run_kind: str,
) -> dict[str, Any]:
    path = repository_root / f"outputs/private/run_status/{status_id}/{state}.json"
    payload = _read_json(path, f"job-{status_id} {state} sentinel")
    expected_exit = 0 if state == "COMPLETE" else 1
    if (
        payload.get("state") != state
        or int(payload.get("exit_status", -1)) != expected_exit
        or str(payload.get("job_id")) != status_id
        or payload.get("git_commit") != source_git_commit
        or payload.get("run_kind") != run_kind
    ):
        raise ContractError(f"Job {status_id} {state} sentinel failed exact provenance checks.")
    opposite = "FAILED" if state == "COMPLETE" else "COMPLETE"
    if (path.parent / f"{opposite}.json").exists():
        raise ContractError(f"Job {status_id} has contradictory final sentinels.")
    return {"status_id": status_id, "sha256": file_sha256(path)}


def _load_recovery_authorization(repository_root: Path, path: Path) -> dict[str, Any]:
    root = repository_root.resolve()
    resolved = path if path.is_absolute() else root / path
    resolved = resolved.resolve()
    expected_root = (root / "configs/recovery").resolve()
    if not resolved.is_relative_to(expected_root) or not resolved.is_file():
        raise ContractError("Fit recovery authorization must be a repository-local configs/recovery YAML file.")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("Fit recovery authorization must be a YAML mapping.")
    return payload


def _require_no_fit_outputs(spec: FinalM1Authorization, repository_root: Path) -> None:
    forbidden = (
        spec.private_root / "fit" / "validation_predictions.parquet",
        spec.aggregate_root / "final_m1_validation.json",
        repository_root / "outputs/private/executed_notebooks/06_final_m1_results.ipynb",
    )
    existing = [str(path.relative_to(repository_root)) for path in forbidden if path.exists()]
    if existing:
        raise ContractError(f"Fit-only recovery refuses existing partial/final outputs: {existing}")


def load_and_validate_fit_inputs(
    *,
    repository_root: Path,
    spec: FinalM1Authorization,
    current_source_git_commit: str,
    recovery_authorization_path: Path | None = None,
    expected_failed_sentinel_sha256: str | None = None,
) -> ValidatedFitInputs:
    """Validate scientific inputs and exact cross-commit recovery provenance before fitting."""
    if FULL_COMMIT.fullmatch(current_source_git_commit) is None:
        raise ContractError("Final fit requires a full current source Git commit.")
    _require_no_fit_outputs(spec, repository_root)
    foundation_path = spec.aggregate_root / "foundation.json"
    throughput_path = spec.aggregate_root / "throughput.json"
    embedding_paths = {
        task: spec.aggregate_root / "embedding" / f"{task}.json" for task in spec.tasks
    }
    foundation = _read_json(foundation_path, "foundation aggregate")
    throughput = _read_json(throughput_path, "throughput aggregate")
    embedding_reports = {
        task: _read_json(path, f"{task} embedding aggregate")
        for task, path in embedding_paths.items()
    }
    config_hash = stable_hash(spec.raw)
    upstream_commit = str(throughput.get("source_git_commit", ""))
    if FULL_COMMIT.fullmatch(upstream_commit) is None:
        raise ContractError("Final throughput aggregate lacks a full source Git commit.")
    foundation_provenance = validate_foundation_reuse(
        repository_root=repository_root,
        spec=spec,
        foundation=foundation,
        current_source_git_commit=upstream_commit,
    )
    if (
        foundation.get("status") != "PASS"
        or foundation.get("content_matching_passed") is not True
        or foundation.get("test_predictions_or_performance_used") is not False
        or throughput.get("status") != "PASS"
        or throughput.get("experiment_id") != spec.experiment_id
        or throughput.get("experiment_config_hash") != config_hash
        or throughput.get("foundation_provenance") != foundation_provenance
        or throughput.get("outcomes_or_performance_read") is not False
        or throughput.get("test_predictions_or_performance_used") is not False
    ):
        raise ContractError("Final fit foundation or throughput provenance failed.")
    if set(embedding_reports) != set(FINAL_TASKS) or any(
        report.get("status") != "PASS"
        or report.get("experiment_id") != spec.experiment_id
        or report.get("experiment_config_hash") != config_hash
        or report.get("source_git_commit") != upstream_commit
        or report.get("task") != task
        or report.get("condition_level_caches") is not True
        or report.get("immutable_verified_cache_reuse") is not True
        or report.get("pilot_cache_reused") is not False
        or report.get("metadata_mirrors_every_embedding_row") is not True
        or report.get("test_split_used") is not False
        for task, report in embedding_reports.items()
    ):
        raise ContractError("A final task embedding report failed exact identity, cache, or split gates.")

    aggregate_hashes = {
        "foundation": file_sha256(foundation_path),
        "throughput": file_sha256(throughput_path),
        "embedding_by_task": {
            task: file_sha256(path) for task, path in embedding_paths.items()
        },
    }
    if upstream_commit == current_source_git_commit:
        if recovery_authorization_path is not None or expected_failed_sentinel_sha256 is not None:
            raise ContractError("Same-source final fitting must not claim fit-recovery authorization.")
        provenance = {
            "mode": "same_source_downstream",
            "upstream_source_git_commit": upstream_commit,
            "fit_source_git_commit": current_source_git_commit,
            "experiment_config_hash": config_hash,
            "foundation_provenance": foundation_provenance,
            "aggregate_sha256": aggregate_hashes,
            "expensive_stages_rerun": False,
            "test_predictions_or_performance_used": False,
        }
        return ValidatedFitInputs(foundation, throughput, embedding_reports, provenance)

    if recovery_authorization_path is None:
        raise ContractError("Cross-commit final fitting requires an explicit fit-only recovery authorization.")
    authorization = _load_recovery_authorization(repository_root, recovery_authorization_path)
    source_jobs = authorization.get("source_jobs")
    controls = authorization.get("controls")
    if (
        authorization.get("version") != 1
        or authorization.get("mode") != "FIT_ONLY_RECOVERY"
        or authorization.get("experiment_id") != spec.experiment_id
        or authorization.get("experiment_config_hash") != config_hash
        or authorization.get("upstream_source_git_commit") != upstream_commit
        or not isinstance(source_jobs, dict)
        or not isinstance(controls, dict)
        or controls
        != {
            "preserve_source_failed_sentinel": True,
            "rerun_foundation": False,
            "rerun_throughput": False,
            "rerun_embeddings": False,
            "require_no_partial_fit_artifacts": True,
            "test_predictions_or_performance_used": False,
            "m2_or_m3_run": False,
        }
    ):
        raise ContractError("Fit-only recovery authorization drifted from its exact contract.")
    foundation_job = source_jobs.get("foundation", {})
    throughput_job = source_jobs.get("throughput", {})
    embedding_job = source_jobs.get("embedding_array", {})
    failed_fit_job = source_jobs.get("failed_fit", {})
    expected_foundation_commit = str(foundation_provenance.get("foundation_source_git_commit", ""))
    if (
        foundation_job
        != {"job_id": 437, "source_git_commit": expected_foundation_commit}
        or throughput_job != {"job_id": 444, "source_git_commit": upstream_commit}
        or embedding_job
        != {
            "job_id": 445,
            "indices": [0, 1, 2, 3],
            "source_git_commit": upstream_commit,
        }
        or failed_fit_job != {"job_id": 446, "source_git_commit": upstream_commit}
    ):
        raise ContractError("Fit-only recovery source jobs or commits drifted.")

    foundation_status = _validate_status(
        repository_root,
        status_id="437",
        state="COMPLETE",
        source_git_commit=expected_foundation_commit,
        run_kind="final_foundation",
    )
    throughput_status = _validate_status(
        repository_root,
        status_id="444",
        state="COMPLETE",
        source_git_commit=upstream_commit,
        run_kind="final_throughput",
    )
    embedding_statuses = {
        str(index): _validate_status(
            repository_root,
            status_id=f"445_{index}",
            state="COMPLETE",
            source_git_commit=upstream_commit,
            run_kind="final_m1_embed_task",
        )["sha256"]
        for index in range(4)
    }
    failed_fit_status = _validate_status(
        repository_root,
        status_id="446",
        state="FAILED",
        source_git_commit=upstream_commit,
        run_kind="final_m1_fit_validation",
    )
    if (
        expected_failed_sentinel_sha256 is None
        or not re.fullmatch(r"[0-9a-f]{64}", expected_failed_sentinel_sha256)
        or failed_fit_status["sha256"] != expected_failed_sentinel_sha256
    ):
        raise ContractError("Job 446 FAILED sentinel hash does not match the submitted recovery evidence.")
    provenance = {
        "mode": "validated_fit_only_recovery",
        "upstream_source_git_commit": upstream_commit,
        "fit_source_git_commit": current_source_git_commit,
        "experiment_config_hash": config_hash,
        "foundation_provenance": foundation_provenance,
        "source_jobs": {
            "foundation": {**foundation_job, "complete_sentinel_sha256": foundation_status["sha256"]},
            "throughput": {**throughput_job, "complete_sentinel_sha256": throughput_status["sha256"]},
            "embedding_array": {**embedding_job, "complete_sentinel_sha256_by_index": embedding_statuses},
            "failed_fit": {
                **failed_fit_job,
                "failed_sentinel_sha256": failed_fit_status["sha256"],
                "preserved_unchanged": True,
            },
        },
        "aggregate_sha256": aggregate_hashes,
        "expensive_stages_rerun": False,
        "test_predictions_or_performance_used": False,
    }
    return ValidatedFitInputs(foundation, throughput, embedding_reports, provenance)
