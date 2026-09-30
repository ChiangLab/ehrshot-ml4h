"""Fail-closed M2 selection-v3 recovery and reference production."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23 import structured_production as legacy
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_selection_v3 as generation
from temporal_ehr.final_m23 import structured_v2_recovery as engineering
from temporal_ehr.final_m23.m2 import balanced_fidelity_sample
from temporal_ehr.final_m23.structured_summary import SummaryContractError
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.pilot.model import load_model_identity
from temporal_ehr.serialization.integrity import stable_hash

VERSION = "m2_selection_v3_reference_v1"
PRIVATE = Path("outputs/private/final") / VERSION
PUBLIC = Path("reports/generated/final") / VERSION
ENGINEERING_PUBLIC = Path("reports/generated/engineering/m2_selection_v3")
RECOVERY_CONFIG = Path("configs/recovery/m2_selection_v3_after_993.yaml")
SOURCE_VERSION = legacy.VERSION
SOURCE_COMMIT = "6b1286b6c5b0f1928b1c5ff7a4b02453c26cc847"
SOURCE_PLAN_HASH = "41613a8075e572fadc1ae9510bf953545752699601b26a2393878fbcbe5520c5"
SOURCE_ARRAY = "993"
SOURCE_SHARDS = 553
SOURCE_ROOT_JOB = "993_0"
SOURCE_ROOT_SHARD = 0
CHUNK_SIZE = 16
GATE_PER_TASK = 8
C1 = legacy.C1


def persist(path: Path, value: dict[str, Any]) -> None:
    """Write once; semantic mismatch is always an error."""
    if path.exists():
        if old.read(path) != value:
            raise ContractError("Immutable selection-v3 artifact differs.")
    else:
        old.write_once(path, value)


def checked_file(root: Path, relative: str, digest: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to((root / PRIVATE).resolve()):
        raise ContractError("Selection-v3 artifact escaped its private namespace.")
    if not path.is_file() or file_sha256(path) != digest:
        raise ContractError("Selection-v3 artifact is missing or changed.")
    return path


def contract(root: Path) -> dict[str, Any]:
    return {
        "version": VERSION,
        "generation_version": generation.VERSION,
        "source_version": SOURCE_VERSION,
        "source_failure_array": SOURCE_ARRAY,
        "source_failure_commit": SOURCE_COMMIT,
        "source_plan_hash": SOURCE_PLAN_HASH,
        "recovery_config_sha256": file_sha256(root / RECOVERY_CONFIG),
        "scope": "C1_train_validation_plus_frozen_C3_manual_sample",
        "clinical_materialization": "exact_source_position_selection",
        "chunk_size": CHUNK_SIZE,
        "active_shards": 2,
        "requests_per_shard": 2,
        "attempts_total_per_key": 2,
        "automatic_repair": False,
        "legacy_cache_reuse": False,
        "embedding_repository": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_tokens": 8192,
        "embedding_dimension": 1024,
        "embedding_batch_size": 8,
        "manual_review_required": True,
        "test_data_used": False,
    }


def split_work(items: dict[str, Any]) -> list[list[str]]:
    keys = sorted(items)
    return [keys[index:index + CHUNK_SIZE] for index in range(0, len(keys), CHUNK_SIZE)]


def source_rows(root: Path):
    """Reuse canonical M1 source construction, never legacy generated summaries."""
    yield from legacy.source_rows(root)


def _sentinel_state(root: Path, job: str) -> str:
    status = root / "outputs/private/run_status" / job
    complete = status / "COMPLETE.json"
    failed = status / "FAILED.json"
    if complete.exists() and failed.exists():
        return "CONFLICT"
    if failed.exists():
        value = old.read(failed)
        return "FAILED" if (
            value.get("state") == "FAILED"
            and value.get("exit_status") != 0
            and value.get("git_commit") == SOURCE_COMMIT
        ) else "INVALID"
    if complete.exists():
        return "COMPLETE"
    return "MISSING"


def source_failure_evidence(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Validate job 993 without printing or copying governed content."""
    plan = legacy.load_plan(root, SOURCE_COMMIT)
    if stable_hash(plan) != SOURCE_PLAN_HASH or plan.get("shards") != SOURCE_SHARDS:
        raise ContractError("Job-993 source plan identity changed.")
    halt_path = root / legacy.PRIVATE / "HALT.json"
    halt = old.read(halt_path)
    if halt != {
        "reason": "rejected_input",
        "job_id": SOURCE_ROOT_JOB,
        "shard": SOURCE_ROOT_SHARD,
        "plan_hash": SOURCE_PLAN_HASH,
    }:
        raise ContractError("Job-993 circuit-breaker evidence changed.")
    status_paths = [
        root / "outputs/private/run_status" / f"{SOURCE_ARRAY}_{index}" / "FAILED.json"
        for index in range(SOURCE_SHARDS)
    ]
    states = Counter(
        _sentinel_state(root, f"{SOURCE_ARRAY}_{index}")
        for index in range(SOURCE_SHARDS)
    )
    if states != {"FAILED": SOURCE_SHARDS}:
        raise ContractError("Job-993 final sentinel inventory changed.")
    failed_items: dict[str, dict[str, Any]] = {}
    categories: Counter[str] = Counter()
    evidence_files = [halt_path, *status_paths]
    generations = root / legacy.PRIVATE / "generations"
    for directory in generations.iterdir():
        assessment_path = directory / "assessment.json"
        if not assessment_path.is_file():
            continue
        assessment = old.read(assessment_path)
        if assessment.get("status") != "FAIL":
            continue
        item = old.read(directory / "source.json")
        request = old.read(directory / "request.json")
        if request != legacy.generation.payload(item["source"], config):
            raise ContractError("Rejected job-993 request no longer matches its source.")
        key = legacy.generation.identity(item, config, plan["served_identity"])
        if key != directory.name:
            raise ContractError("Rejected job-993 cache identity changed.")
        failed_items[stable_hash(item)] = item
        categories.update(map(str, assessment.get("errors", [])))
        evidence_files.extend(path for path in directory.glob("*.json") if path.is_file())
    expected = {
        "duplicate_series",
        "identical_rejected_content",
        "invented_changed_or_reordered_series",
    }
    if len(failed_items) != 2 or set(categories) != expected:
        raise ContractError("Job-993 rejected-input evidence changed.")
    return {
        "plan": plan,
        "failed_items": failed_items,
        "evidence_hash": stable_hash({
            str(path.relative_to(root)): file_sha256(path)
            for path in sorted(evidence_files)
        }),
        "failure_categories": {
            key: value if value >= 20 else "<20"
            for key, value in sorted(categories.items())
        },
        "failed_inputs": "<20",
        "all_array_tasks_failed": SOURCE_SHARDS,
    }


def _fact_count(item: dict[str, Any]) -> int:
    return sum(len(window["facts"]) for window in item["source"]["windows"])


def _even_sample(items: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    if len(items) < count:
        raise ContractError("Insufficient training-only inputs for independent M2 gates.")
    ordered = sorted(items, key=lambda row: (
        _fact_count(row["item"]), stable_hash({"salt": salt, "item": row["item"]})
    ))
    positions = [min(len(ordered) - 1, ((2 * i + 1) * len(ordered)) // (2 * count)) for i in range(count)]
    selected = [ordered[index] for index in positions]
    if len({stable_hash(row["item"]) for row in selected}) != count:
        raise ContractError("Engineering sample selection was not unique.")
    return selected


def engineering_samples(
    rows_by_task: dict[str, list[dict[str, Any]]],
    failed_items: dict[str, dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Outcome-independent gates: include old failures, then independent train C1."""
    item_task: dict[str, str] = {}
    for task, rows in rows_by_task.items():
        if task.startswith("_"):
            continue
        for row in rows:
            key = stable_hash(row["item"])
            item_task.setdefault(key, task)
    if any(key not in item_task for key in failed_items):
        raise ContractError("A rejected job-993 input is absent from canonical source rows.")
    canary: dict[str, dict[str, Any]] = {}
    confirmation: dict[str, dict[str, Any]] = {}
    excluded: set[str] = set()
    for task in old.TASKS:
        task_failed = [item for key, item in failed_items.items() if item_task[key] == task]
        if len(task_failed) > GATE_PER_TASK:
            raise ContractError("Too many prior failures for the bounded canary.")
        candidates = [row for row in rows_by_task[task]
                      if row["split"] == "train" and row["condition"] == C1]
        failed_hashes = {stable_hash(item) for item in task_failed}
        candidates = [row for row in candidates if stable_hash(row["item"]) not in failed_hashes]
        fill = _even_sample(candidates, GATE_PER_TASK - len(task_failed), f"canary:{task}")
        selected = task_failed + [row["item"] for row in fill]
        if len(selected) != GATE_PER_TASK:
            raise ContractError("Canary task allocation changed.")
        for item in selected:
            key = stable_hash(item)
            if key in excluded:
                raise ContractError("Canary contains a duplicate input.")
            excluded.add(key)
            canary[generation.identity(item, rows_by_task["_config"], rows_by_task["_served"])] = item
        remaining = [row for row in candidates if stable_hash(row["item"]) not in excluded]
        for row in _even_sample(remaining, GATE_PER_TASK, f"confirmation:{task}"):
            item = row["item"]
            key = stable_hash(item)
            excluded.add(key)
            confirmation[generation.identity(
                item, rows_by_task["_config"], rows_by_task["_served"]
            )] = item
    if len(canary) != 32 or len(confirmation) != 32:
        raise ContractError("Engineering gates must contain 32 unique inputs each.")
    return {"canary": canary, "confirmation": confirmation}


def freeze_plan(root: Path, model_root: Path, commit: str) -> dict[str, Any]:
    frozen_config = old.load_config(root)
    evidence = source_failure_evidence(root, frozen_config)
    config, _, _ = engineering.environment(root, model_root)
    if config != frozen_config:
        raise ContractError("Selection-v3 generation configuration changed.")
    old_plan = evidence["plan"]
    if old.tokenizer_fingerprint(model_root) != old_plan["tokenizer_hash"]:
        raise ContractError("Tokenizer differs from job-993 source evidence.")
    identity = load_model_identity(
        model_root / "Qwen3-Embedding-0.6B",
        expected_repository="Qwen/Qwen3-Embedding-0.6B",
    )
    client = old.PrivateClient(config)
    if client.identity() != old_plan["served_identity"]:
        raise ContractError("Served instruction-model identity changed.")

    files: dict[str, str] = {}
    items: dict[str, dict[str, Any]] = {}
    review_records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    for task, records in source_rows(root):
        raw_rows[task] = []
        counts[task] = len(records)
        for original in records:
            row = dict(original)
            item = row.pop("item")
            key = generation.identity(item, config, old_plan["served_identity"])
            if key in items and items[key] != item:
                raise ContractError("One v3 identity maps to different inputs.")
            items[key] = item
            row["generation_key"] = key
            raw_rows[task].append({**row, "item": item})
            review_records.append(row)
        frozen_rows = [{k: v for k, v in row.items() if k != "item"} for row in raw_rows[task]]
        path = root / PRIVATE / "plan" / f"{task}.json"
        persist(path, {"rows": frozen_rows})
        files[str(path.relative_to(root))] = file_sha256(path)

    gate_context: dict[str, Any] = {**raw_rows, "_config": config, "_served": old_plan["served_identity"]}
    gates = engineering_samples(gate_context, evidence["failed_items"])
    for phase, gate_items in gates.items():
        path = root / PRIVATE / "plan" / f"gate-{phase}.json"
        persist(path, {"items": gate_items})
        files[str(path.relative_to(root))] = file_sha256(path)

    chunks = split_work(items)
    if not chunks or len(chunks) > 999:
        raise ContractError("Selection-v3 worklist exceeds the bounded array design.")
    for index, keys in enumerate(chunks):
        path = root / PRIVATE / "plan" / f"shard-{index}.json"
        persist(path, {"items": {key: items[key] for key in keys}})
        files[str(path.relative_to(root))] = file_sha256(path)

    manual = balanced_fidelity_sample(review_records, histories_per_task_condition=5)
    manual_keys = [
        {k: row[k] for k in ("task", "condition", "instance_key", "generation_key")}
        for row in manual
    ]
    plan = {
        "version": VERSION,
        "execution_commit": commit,
        "contract": contract(root),
        "source_failure_evidence_hash": evidence["evidence_hash"],
        "served_identity": old_plan["served_identity"],
        "tokenizer_hash": old_plan["tokenizer_hash"],
        "model_artifact_hash": identity.artifact_hash,
        "model_tokenizer_hash": identity.tokenizer_hash,
        "files": files,
        "manual_sample": manual_keys,
        "shards": len(chunks),
        "unique_inputs": len(items),
        "rows_by_task": counts,
        "gate_sizes": {phase: len(value) for phase, value in gates.items()},
        "test_data_used": False,
    }
    persist(root / PRIVATE / "PLAN.json", plan)
    report = {
        "version": VERSION,
        "status": "PASS",
        "phase": "plan",
        "source_commit": commit,
        "source_failure_commit": SOURCE_COMMIT,
        "source_failure_array": SOURCE_ARRAY,
        "source_failure_evidence_hash": evidence["evidence_hash"],
        "source_failure_categories": evidence["failure_categories"],
        "plan_hash": stable_hash(plan),
        "shards": len(chunks),
        "unique_inputs": len(items),
        "rows_by_task": counts,
        "gate_sizes": plan["gate_sizes"],
        "clinical_materialization": "exact_source_position_selection",
        "legacy_cache_rows_reused": 0,
        "test_data_used": False,
        "scientific_M2_complete": False,
    }
    assert_privacy_safe_report(json.dumps(report))
    persist(root / PUBLIC / "plan.json", report)
    return report


def load_plan(root: Path, commit: str) -> dict[str, Any]:
    plan = old.read(root / PRIVATE / "PLAN.json")
    report = old.read(root / PUBLIC / "plan.json")
    if (
        plan.get("execution_commit") != commit
        or plan.get("contract") != contract(root)
        or report.get("plan_hash") != stable_hash(plan)
        or report.get("status") != "PASS"
        or plan.get("test_data_used") is not False
    ):
        raise ContractError("Selection-v3 plan/source/contract mismatch.")
    return plan


def planned_file(root: Path, plan: dict[str, Any], name: str) -> dict[str, Any]:
    relative = str(PRIVATE / "plan" / name)
    return old.read(checked_file(root, relative, plan["files"][relative]))


def verified_row(
    root: Path,
    plan: dict[str, Any],
    key: str,
    item: dict[str, Any],
    env: tuple[Any, Any, Any],
) -> dict[str, Any]:
    directory = root / PRIVATE / "generations" / key
    seal = old.read(directory / "SEALED.json")
    files = {
        path.name: file_sha256(path)
        for path in directory.glob("*.json")
        if path.name != "SEALED.json"
    }
    if seal != {"files": files, "plan_hash": stable_hash(plan)}:
        raise ContractError("Selection-v3 evidence hash or plan binding changed.")
    result = generation.verify_item(
        directory,
        item,
        *env,
        plan["served_identity"],
        plan["tokenizer_hash"],
    )
    if key != result["generation_key"]:
        raise ContractError("Selection-v3 cache key mismatch.")
    return result


def generate_row(
    root: Path,
    plan: dict[str, Any],
    key: str,
    item: dict[str, Any],
    env: tuple[Any, Any, Any],
    client: Any = None,
) -> dict[str, Any]:
    """Resume saved responses; deterministic source materialization is always rechecked."""
    directory = root / PRIVATE / "generations" / key
    if (directory / "SEALED.json").exists():
        return verified_row(root, plan, key, item, env)
    config, thresholds, counter = env
    if generation.identity(item, config, plan["served_identity"]) != key:
        raise ContractError("Wrong selection-v3 request identity.")
    persist(directory / "source.json", item)
    persist(directory / "request.json", generation.payload(item["source"], config))
    client = client or old.PrivateClient(config)
    if (directory / "assessment.json").exists():
        return old.read(directory / "assessment.json")
    errors: list[str] = []
    rejected: set[str] = set()
    result: dict[str, Any] = {"status": "FAIL", "errors": errors}
    for attempt in (1, 2):
        response_path = directory / f"response-{attempt}.json"
        transport_path = directory / f"transport-{attempt}.json"
        started_path = directory / f"started-{attempt}.json"
        try:
            if transport_path.exists():
                raise RuntimeError("Previously recorded transport failure")
            if not response_path.exists():
                if started_path.exists():
                    persist(transport_path, {"error": "interrupted_request"})
                    raise RuntimeError("Interrupted request")
                persist(started_path, {"attempt": attempt})
                try:
                    response = client.request(
                        "chat/completions",
                        generation.payload(item["source"], config),
                    )
                except RuntimeError:
                    persist(transport_path, {"error": "endpoint_failure"})
                    raise
                persist(response_path, response)
            selection = old.response_document(old.read(response_path))
            digest = stable_hash(selection)
            if digest in rejected:
                raise SummaryContractError("identical_rejected_selection")
            rejected.add(digest)
            document, validation, summary = generation.validate(
                selection,
                item["source"],
                thresholds=thresholds,
                token_counter=counter,
            )
            result = {
                "status": "PASS",
                "selection": selection,
                "document": document,
                "summary": summary,
                "validation": validation,
                "attempts": attempt,
                "errors": errors,
                "generation_key": key,
                "embedding_input_key": generation.renderer.embedding_identity(
                    key, summary, plan["tokenizer_hash"]
                ),
            }
            break
        except SummaryContractError as exc:
            errors.append(str(exc))
        except old.NonRetryableSummaryError:
            errors.append("embedding_token_budget")
        except RuntimeError:
            errors.append("endpoint_failure")
    persist(directory / "assessment.json", result)
    if result["status"] == "PASS":
        generation.verify_item(
            directory,
            item,
            config,
            thresholds,
            counter,
            plan["served_identity"],
            plan["tokenizer_hash"],
        )
        persist(directory / "SEALED.json", {
            "plan_hash": stable_hash(plan),
            "files": {
                path.name: file_sha256(path)
                for path in directory.glob("*.json")
                if path.name != "SEALED.json"
            },
        })
    return result


def _run_items(
    root: Path,
    plan: dict[str, Any],
    items: dict[str, dict[str, Any]],
    env: tuple[Any, Any, Any],
    *,
    workers: int,
    event: str,
    label: str,
) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    client = old.PrivateClient(env[0])
    if client.identity() != plan["served_identity"]:
        raise ContractError("Endpoint model identity changed before generation.")
    entries = list(items.items())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for offset in range(0, len(entries), workers):
            wave = list(pool.map(
                lambda pair: generate_row(root, plan, *pair, env, client),
                entries[offset:offset + workers],
            ))
            results.extend(wave)
            print(json.dumps({
                "event": event,
                "label": label,
                "checked": len(results),
                "planned": len(items),
                "elapsed_seconds": round(time.monotonic() - started),
            }), flush=True)
    if client.identity() != plan["served_identity"]:
        raise ContractError("Endpoint model identity changed during generation.")
    return results, time.monotonic() - started


def run_gate(
    root: Path,
    model_root: Path,
    commit: str,
    phase: str,
    job: str,
) -> dict[str, Any]:
    if phase not in {"canary", "confirmation"}:
        raise ContractError("Invalid selection-v3 gate phase.")
    plan = load_plan(root, commit)
    items = planned_file(root, plan, f"gate-{phase}.json")["items"]
    env = engineering.environment(root, model_root)
    results, runtime = _run_items(
        root, plan, items, env, workers=4, event="m2_selection_gate_progress", label=phase
    )
    failures = Counter(
        error
        for result in results
        if result.get("status") != "PASS"
        for error in result.get("errors", [])
    )
    passed = len(results) == len(items) and not failures
    report = {
        "version": VERSION,
        "status": "PASS" if passed else "FAIL",
        "phase": phase,
        "job_id": job,
        "source_commit": commit,
        "plan_hash": stable_hash(plan),
        "inputs_checked": len(results),
        "inputs_accepted": sum(result.get("status") == "PASS" for result in results),
        "failure_categories": {
            key: value if value >= 20 else "<20" for key, value in sorted(failures.items())
        },
        "clinical_materialization": "exact_source_position_selection",
        "runtime_seconds": runtime,
        "test_data_used": False,
        "scientific_M2_complete": False,
    }
    assert_privacy_safe_report(json.dumps(report))
    persist(root / ENGINEERING_PUBLIC / f"{phase}-{job}.json", report)
    if not passed:
        raise ContractError(f"Selection-v3 {phase} gate rejected an input.")
    return report


def verify_gates(
    root: Path,
    model_root: Path,
    commit: str,
    canary_job: str,
    confirmation_job: str,
    job: str,
) -> dict[str, Any]:
    """Read-only recursive gate verifier; this is the sole production dependency."""
    plan = load_plan(root, commit)
    env = engineering.environment(root, model_root)
    reports: dict[str, dict[str, Any]] = {}
    all_keys: set[str] = set()
    minimum: dict[str, float] = {}
    for phase, parent in (("canary", canary_job), ("confirmation", confirmation_job)):
        old._sentinel(root, parent, "COMPLETE", commit)
        report = old.read(root / ENGINEERING_PUBLIC / f"{phase}-{parent}.json")
        items = planned_file(root, plan, f"gate-{phase}.json")["items"]
        required = {
            "version": VERSION,
            "status": "PASS",
            "phase": phase,
            "job_id": parent,
            "source_commit": commit,
            "plan_hash": stable_hash(plan),
            "inputs_checked": len(items),
            "inputs_accepted": len(items),
            "clinical_materialization": "exact_source_position_selection",
            "test_data_used": False,
            "scientific_M2_complete": False,
        }
        if any(report.get(key) != value for key, value in required.items()):
            raise ContractError(f"Selection-v3 {phase} report is incomplete or stale.")
        if all_keys.intersection(items):
            raise ContractError("Canary and confirmation gates overlap.")
        all_keys.update(items)
        for key, item in items.items():
            result = verified_row(root, plan, key, item, env)
            for name, value in result["validation"]["retention"].items():
                minimum[name] = min(minimum.get(name, 1.0), float(value))
        reports[phase] = report
    readiness = {
        "version": VERSION,
        "status": "PASS",
        "phase": "readiness",
        "job_id": job,
        "source_commit": commit,
        "plan_hash": stable_hash(plan),
        "canary_job_id": canary_job,
        "confirmation_job_id": confirmation_job,
        "gate_report_hashes": {key: stable_hash(value) for key, value in reports.items()},
        "inputs_checked": len(all_keys),
        "inputs_accepted": len(all_keys),
        "minimum_retention": minimum,
        "clinical_materialization": "exact_source_position_selection",
        "production_authorized": True,
        "test_data_used": False,
        "scientific_M2_complete": False,
    }
    assert_privacy_safe_report(json.dumps(readiness))
    persist(root / ENGINEERING_PUBLIC / f"readiness-{job}.json", readiness)
    return readiness


def verify_readiness(
    root: Path,
    model_root: Path,
    commit: str,
    readiness_job: str,
) -> dict[str, Any]:
    """Reconstruct readiness instead of trusting a dependency or file alone."""
    old._sentinel(root, readiness_job, "COMPLETE", commit)
    report = old.read(root / ENGINEERING_PUBLIC / f"readiness-{readiness_job}.json")
    verified = verify_gates(
        root,
        model_root,
        commit,
        str(report.get("canary_job_id", "")),
        str(report.get("confirmation_job_id", "")),
        readiness_job,
    )
    if verified != report or report.get("production_authorized") is not True:
        raise ContractError("Selection-v3 readiness evidence changed.")
    return report


def run_shard(root: Path, model_root: Path, commit: str, index: int, job: str) -> dict[str, Any]:
    plan = load_plan(root, commit)
    if (root / PRIVATE / "HALT.json").exists():
        raise ContractError("Selection-v3 circuit breaker is set; preserve and review evidence.")
    if not 0 <= index < plan["shards"]:
        raise ContractError("Invalid selection-v3 shard index.")
    items = planned_file(root, plan, f"shard-{index}.json")["items"]
    if not 0 < len(items) <= CHUNK_SIZE:
        raise ContractError("Unbounded or empty selection-v3 shard.")
    env = engineering.environment(root, model_root)
    results, runtime = _run_items(
        root,
        plan,
        items,
        env,
        workers=2,
        event="m2_selection_production_progress",
        label=str(index),
    )
    passed = len(results) == len(items) and all(result["status"] == "PASS" for result in results)
    report = {
        "version": VERSION,
        "status": "PASS" if passed else "FAIL",
        "phase": "shard",
        "job_id": job,
        "shard": index,
        "plan_hash": stable_hash(plan),
        "source_commit": commit,
        "all_inputs_accepted": passed,
        "test_data_used": False,
        "scientific_M2_complete": False,
        "runtime_seconds": runtime,
    }
    persist(root / PUBLIC / "attempts" / f"shard-{index}-{job}.json", report)
    if not passed:
        with suppress(FileExistsError):
            old.write_once(root / PRIVATE / "HALT.json", {
                "reason": "rejected_selection",
                "job_id": job,
                "shard": index,
                "plan_hash": stable_hash(plan),
            })
        raise ContractError("Selection-v3 shard rejected an input; preserve evidence.")
    return report
