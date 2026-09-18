"""Job-596 recovery: small train-only engineering gate, never a clinical fit."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import load_final_m1_authorization
from temporal_ehr.final_m1.embedding import condition_label
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.final_m23.m2 import (
    LocalSummaryTokenValidator,
    NonRetryableSummaryError,
    OnPremiseSummaryClient,
    m2_source_text,
    validate_bounded_summary,
)
from temporal_ehr.final_m23.protocol import (
    load_final_m23_authorization,
    require_prompt4_hard_gates,
)
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.serialization.integrity import stable_hash

SOURCE_596 = "0d17bac55a561880086aae560640bfcdb719f021"
SOURCE_M3 = "1a97d8f1a70c47e18f96a2f56e63a370d6ff9913"
SCHEMA = "m2_after596_singleton_canary_v1"
CONDITIONS = ("C1_correct_compact_time", "C3_shuffled_earlier_time_labels:seed=20260815")
REPORT_ROOT = Path("reports/generated/engineering/m2_after596")
PRIVATE_ROOT = Path("outputs/private/recovery/m2_after596")


def failure_category(error: Exception) -> str:
    """Return only an allow-listed category; never copy API text into public reports."""
    message = str(error).lower()
    for needle, category in (
        ("invalid recurrence count", "recurrence_syntax"),
        ("recurrence", "recurrence_mismatch_or_retention"),
        ("trend", "trend"),
        ("association", "association"),
        ("unsupported code", "unsupported_fact"),
        ("time", "timing_or_transport"),
        ("retention", "retention"),
        ("finish", "completion"),
        ("token", "token_budget"),
        ("field", "format"),
        ("prose", "format"),
        ("prefix", "format"),
    ):
        if needle in message:
            return category
    return "endpoint_or_other"


def select_canary_sources(rows: list[dict[str, Any]], instruction: str) -> list[dict[str, str]]:
    """Four length strata per condition, training only; independent of labels/results.

    The frozen 40-history manual validation sample is not used to debug the interface.
    Only supplied prepared C1/C3 histories are used: no new perturbations are invented.
    """
    selected = []
    for condition in CONDITIONS:
        sources = {
            m2_source_text(str(row["text"]), instruction)
            for row in rows
            if row["split"] == "train" and condition_label(row) == condition
        }
        ranked = sorted(sources, key=lambda text: (len(text), stable_hash(text)))
        if len(ranked) < 4:
            raise ContractError("Canary requires four distinct training inputs per condition.")
        for index in (0, (len(ranked) - 1) // 3, 2 * (len(ranked) - 1) // 3, len(ranked) - 1):
            selected.append({"condition": condition, "source_text": ranked[index]})
    return selected


def _sentinel(root: Path, job: str, state: str, commit: str) -> Path:
    directory = root / "outputs/private/run_status" / job
    path = directory / f"{state}.json"
    other = directory / ("FAILED.json" if state == "COMPLETE" else "COMPLETE.json")
    if not path.is_file() or other.exists():
        raise ContractError(f"Missing or conflicting final sentinel for job {job}.")
    data = json.loads(path.read_text())
    if data.get("job_id") != job or data.get("state") != state or data.get("git_commit") != commit:
        raise ContractError(f"Sentinel identity mismatch for job {job}.")
    if state == "COMPLETE" and data.get("exit_status") != 0:
        raise ContractError(f"Nonzero COMPLETE sentinel for job {job}.")
    if state == "FAILED" and (not isinstance(data.get("exit_status"), int) or data["exit_status"] == 0):
        raise ContractError(f"Invalid FAILED exit status for job {job}.")
    return path


def preserved_evidence(root: Path) -> dict[str, str]:
    """Hash old statuses/logs without changing them; missing unstarted shards are allowed."""
    hashes = {}
    failures = 0
    for index in range(768):
        job = f"596_{index}"
        directory = root / "outputs/private/run_status" / job
        for state in ("COMPLETE", "FAILED"):
            if (directory / f"{state}.json").exists():
                _sentinel(root, job, state, SOURCE_596)
                failures += state == "FAILED"
        for path in sorted(directory.glob("*.json")):
            hashes[str(path.relative_to(root))] = file_sha256(path)
    if failures == 0:
        raise ContractError("No verified job-596 failure evidence; wrong recovery target.")
    for job in [*(f"597_{i}" for i in range(4)), "598"]:
        directory = root / "outputs/private/run_status" / job
        if (directory / "COMPLETE.json").exists():
            raise ContractError("Old downstream completion requires separate provenance review.")
        for path in sorted(directory.glob("*.json")):
            hashes[str(path.relative_to(root))] = file_sha256(path)
    for pattern in ("slurm-*-596_*.out", "slurm-*-597_*.out", "slurm-*-598.out"):
        for path in sorted(root.glob(pattern)):
            hashes[str(path.relative_to(root))] = file_sha256(path)
    for job in [*(f"470_{i}" for i in range(4)), *(f"471_{i}" for i in range(12))]:
        path = _sentinel(root, job, "COMPLETE", SOURCE_M3)
        hashes[str(path.relative_to(root))] = file_sha256(path)
    return hashes


class RecordingClient(OnPremiseSummaryClient):
    """Retain rejected completions privately for source-to-output diagnosis on DGX."""

    def __init__(self, *, directory: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.directory = directory
        self.attempt = 0

    def _request(self, path: str, payload: Any = None) -> dict[str, Any]:
        response = super()._request(path, payload)
        if payload is not None:
            self.attempt += 1
            atomic_json(self.directory / f"response-{self.attempt}.json", response)
        return response


def assess_source(
    client: OnPremiseSummaryClient, item: dict[str, str], m2: dict[str, Any], token_counter: Any, directory: Path
) -> dict[str, Any]:
    """Same prompt/decoding/three-attempt limit, with all rejection categories recorded."""
    atomic_json(directory / "source.json", item)
    errors: Counter[str] = Counter()
    started = time.monotonic()
    for attempt in range(1, int(m2["execution"]["generation_contract_attempts"]) + 1):
        try:
            generation = client.summarize_response(
                system_prompt=m2["prompt"],
                source_text=item["source_text"],
                decoding=m2["decoding"],
            )
            validation = validate_bounded_summary(
                generation.summary,
                source_text=item["source_text"],
                token_counter=token_counter,
                thresholds=m2["fidelity"],
            )
            result = {
                "status": "PASS",
                "attempts": attempt,
                "errors": dict(errors),
                "explicit_singleton": bool(re.search(r"\boccurrences=1(?:\s|$)", generation.summary)),
                "validation": validation,
                "seconds": time.monotonic() - started,
            }
            atomic_json(directory / "assessment.json", result)
            return result
        except (NonRetryableSummaryError, RuntimeError) as exc:
            errors[failure_category(exc)] += 1
    result = {"status": "FAIL", "errors": dict(errors), "seconds": time.monotonic() - started}
    atomic_json(directory / "assessment.json", result)
    return result


def run_canary(root: Path, model_root: Path, job: str, commit: str) -> dict[str, Any]:
    config = root / "configs/experiments/final_m2_m3.yaml"
    require_prompt4_hard_gates(root, config)
    evidence = preserved_evidence(root)
    spec = load_final_m23_authorization(config, repository_root=root)
    m1 = load_final_m1_authorization(spec.m1_config_path, repository_root=root)
    private = root / PRIVATE_ROOT / job
    private.mkdir(parents=True, exist_ok=False)
    atomic_json(private / "preserved_evidence.json", evidence)
    token_counter = LocalSummaryTokenValidator(model_root / spec.model_directory_name, maximum_tokens=8192)
    client_args = {"api_base": spec.m2["api_base"], "model": spec.m2["instruction_model"], "timeout_seconds": 300}
    identity = OnPremiseSummaryClient(**client_args).model_revision_hash()
    items = []
    for task in spec.tasks:
        rows = read_final_task_texts(m1, task)
        items.extend({"task": task, **item} for item in select_canary_sources(rows, m1.instructions[task]))
    atomic_json(private / "inputs.json", {"items": items})

    def evaluate(pair: tuple[int, dict[str, str]]) -> dict[str, Any]:
        index, item = pair
        directory = private / f"item-{index:02d}"
        return assess_source(
            RecordingClient(directory=directory, **client_args), item, spec.m2, token_counter, directory
        )

    # Only four total outstanding requests, not 32; this is a fidelity gate, not a load test.
    results_by_index = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(evaluate, pair): pair[0] for pair in enumerate(items)}
        for future in as_completed(futures):
            results_by_index[futures[future]] = future.result()
            print(
                json.dumps({"event": "m2_canary_progress", "checked": len(results_by_index), "total": 32}), flush=True
            )
    results = [results_by_index[index] for index in range(32)]
    if evidence != preserved_evidence(root):
        raise ContractError("Original job evidence changed during the canary.")
    if identity != OnPremiseSummaryClient(**client_args).model_revision_hash():
        raise ContractError("Served model identity changed during the canary.")
    errors: Counter[str] = Counter()
    for result in results:
        errors.update(result["errors"])
    passed = len(results) == 32 and all(result["status"] == "PASS" for result in results)
    manifest = {str(path.relative_to(private)): file_sha256(path) for path in sorted(private.rglob("*.json"))}
    atomic_json(private / "manifest.json", manifest)
    report = {
        "schema": SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "job_id": job,
        "source_git_commit": SOURCE_596,
        "recovery_git_commit": commit,
        "config_sha256": file_sha256(config),
        "served_identity_hash": identity,
        "original_evidence_hash": stable_hash(evidence),
        "private_assessments_hash": stable_hash(results),
        "private_manifest_sha256": file_sha256(private / "manifest.json"),
        "sample_design": "train_only_four_length_strata_per_task_per_C1_C3_condition",
        "planned_inputs": 32,
        "all_inputs_passed": passed,
        "failure_categories": {key: (count if count >= 20 else "<20") for key, count in sorted(errors.items())},
        "embedding_model": spec.model_repository,
        "embedding_model_run": False,
        "test_data_used": False,
        "manual_fidelity_review_complete": False,
        "scientific_M2_complete": False,
        "full_run_authorized": False,
        "interpretation": "Engineering gate only; a passing sample does not prove all summaries valid.",
    }
    assert_privacy_safe_report(json.dumps(report))
    atomic_json(root / REPORT_ROOT / f"canary-{job}.json", report)
    return report


def verify_canary(root: Path, job: str, commit: str) -> dict[str, Any]:
    """Never promote existence alone to recovery readiness."""
    _sentinel(root, job, "COMPLETE", commit)
    report = json.loads((root / REPORT_ROOT / f"canary-{job}.json").read_text())
    if any(
        (
            report.get("schema") != SCHEMA,
            report.get("status") != "PASS",
            report.get("job_id") != job,
            report.get("recovery_git_commit") != commit,
            report.get("source_git_commit") != SOURCE_596,
            report.get("planned_inputs") != 32,
            report.get("all_inputs_passed") is not True,
            report.get("test_data_used") is not False,
            report.get("config_sha256") != file_sha256(root / "configs/experiments/final_m2_m3.yaml"),
            report.get("original_evidence_hash") != stable_hash(preserved_evidence(root)),
        )
    ):
        raise ContractError("Canary missing, failed, stale, or provenance-mismatched.")
    private = root / PRIVATE_ROOT / job
    manifest_path = private / "manifest.json"
    if file_sha256(manifest_path) != report.get("private_manifest_sha256"):
        raise ContractError("Canary private manifest hash failed.")
    manifest = json.loads(manifest_path.read_text())
    observed = {
        str(path.relative_to(private)): file_sha256(path)
        for path in sorted(private.rglob("*.json"))
        if path != manifest_path
    }
    if manifest != observed:
        raise ContractError("Canary source/response evidence changed.")
    results = [json.loads((private / f"item-{i:02d}/assessment.json").read_text()) for i in range(32)]
    if (
        not all(result.get("status") == "PASS" for result in results)
        or stable_hash(results) != report["private_assessments_hash"]
    ):
        raise ContractError("Canary private assessment hashes failed.")
    return report


def current_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
