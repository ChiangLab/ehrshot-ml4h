"""Version-isolated M2 reference production; no legacy caches or held-out access."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_render_v2 as renderer
from temporal_ehr.final_m23 import structured_v2 as generation
from temporal_ehr.final_m23 import structured_v2_recovery as engineering
from temporal_ehr.final_m23.m2 import balanced_fidelity_sample, m2_stage_rows
from temporal_ehr.final_m23.structured_summary import SummaryContractError
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.pilot.model import load_model_identity
from temporal_ehr.serialization.integrity import stable_hash

VERSION = "m2_structured_v2_reference_v1"
PRIVATE = Path("outputs/private/final") / VERSION
PUBLIC = Path("reports/generated/final") / VERSION
ENGINEERING_COMMIT = "6c80719266485c861fd8aa13f931eebcf5bfc8f8"
ENGINEERING_PLAN = "c508d24be784462dda204a0cde8b91d3606dac4a6c5db5c65f01d6c84ce676bd"
READINESS_MANIFEST = "187ae0b677740434825d84ee589e3f6bdb6cbbd3327366c4c967deaefddec1c6"
CHUNK_SIZE = 16
C1 = "C1_correct_compact_time"


def persist(path: Path, value: dict) -> None:
    """Idempotent only for byte-independent semantic equality; never replace evidence."""
    if path.exists():
        if old.read(path) != value:
            raise ContractError("Immutable production artifact differs; stop and review.")
    else:
        old.write_once(path, value)


def checked_file(root: Path, relative: str, digest: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to((root / PRIVATE).resolve()):
        raise ContractError("Production artifact escaped its private namespace.")
    if not path.is_file() or file_sha256(path) != digest:
        raise ContractError("Production artifact is missing or changed.")
    return path


def contract(root: Path) -> dict:
    return {"version": VERSION, "engineering_contract": engineering.contract(root),
            "scope": "C1_train_validation_plus_frozen_C3_manual_sample",
            "chunk_size": CHUNK_SIZE, "active_shards": 2, "requests_per_shard": 2,
            "attempts_total_per_key": 2, "automatic_retry_of_rejected_key": False,
            "embedding_repository": "Qwen/Qwen3-Embedding-0.6B", "embedding_tokens": 8192,
            "embedding_dimension": 1024, "embedding_batch_size": 8,
            "manual_review_required": True, "test_data_used": False}


def split_work(items: dict) -> list[list[str]]:
    keys = sorted(items)
    return [keys[i:i + CHUNK_SIZE] for i in range(0, len(keys), CHUNK_SIZE)]


def source_rows(root: Path):
    old.require_prompt4_hard_gates(root, root / old.BASELINE)
    spec = old.load_final_m23_authorization(root / old.BASELINE, repository_root=root)
    m1 = old.load_final_m1_authorization(spec.m1_config_path, repository_root=root)
    for task in old.TASKS:
        rows = old.read_final_task_texts(m1, task)
        selected = m2_stage_rows(rows, spec=spec, stage="reference")
        records = []
        for row in selected:
            if row["split"] not in {"train", "validation"}:
                raise ContractError("Production is train/validation only.")
            records.append({"task": task, "condition": old.condition_label(row),
                "split": row["split"], "instance_key": row["instance_key"], "target": row["target"],
                "source_text": str(row["text"]),
                "item": {"condition": old.condition_label(row),
                         "source": old.ordered_source(str(row["text"]), m1.instructions[task])}})
        if len({(r["split"], r["instance_key"], r["condition"]) for r in records}) != len(records):
            raise ContractError("Duplicate production patient/condition row.")
        yield task, records


def freeze_plan(root: Path, model_root: Path, commit: str) -> dict:
    ready = engineering.verify(root, model_root, "990", "readiness", ENGINEERING_COMMIT)
    if ready["plan_hash"] != ENGINEERING_PLAN or ready["private_manifest_hash"] != READINESS_MANIFEST:
        raise ContractError("Wrong reviewed engineering evidence.")
    config, _, _ = engineering.environment(root, model_root)
    eng_plan = old.read(root / engineering.PRIVATE / "990/plan.json")
    if old.tokenizer_fingerprint(model_root) != eng_plan["tokenizer_hash"]:
        raise ContractError("Tokenizer differs from successful engineering evidence.")
    identity = load_model_identity(model_root / "Qwen3-Embedding-0.6B",
                                   expected_repository="Qwen/Qwen3-Embedding-0.6B")
    client = old.PrivateClient(config)
    if client.identity() != eng_plan["served_identity"]:
        raise ContractError("Served model identity changed since confirmation.")
    base = root / PRIVATE
    files, items, review_records, counts = {}, {}, [], {}
    for task, records in source_rows(root):
        counts[task] = len(records)
        for row in records:
            key = generation.identity(row["item"], config, eng_plan["served_identity"])
            item = row.pop("item")
            if key in items and items[key] != item:
                raise ContractError("One generation identity maps to different frozen inputs.")
            items[key] = item
            row["generation_key"] = key
        path = base / "plan" / f"{task}.json"
        persist(path, {"rows": records})
        files[str(path.relative_to(root))] = file_sha256(path)
        review_records.extend(records)
    manual = balanced_fidelity_sample(review_records, histories_per_task_condition=5)
    review_keys = [{k: r[k] for k in ("task", "condition", "instance_key", "generation_key")}
                   for r in manual]
    chunks = split_work(items)
    if not chunks or len(chunks) > 999:
        raise ContractError("Worklist exceeds the bounded array design; no jobs may be submitted.")
    for index, keys in enumerate(chunks):
        path = base / "plan" / f"shard-{index}.json"
        persist(path, {"items": {key: items[key] for key in keys}})
        files[str(path.relative_to(root))] = file_sha256(path)
    plan = {"version": VERSION, "execution_commit": commit, "contract": contract(root),
        "engineering_commit": ENGINEERING_COMMIT, "engineering_plan_hash": ENGINEERING_PLAN,
        "readiness_hash": stable_hash(ready), "served_identity": eng_plan["served_identity"],
        "tokenizer_hash": eng_plan["tokenizer_hash"], "model_artifact_hash": identity.artifact_hash,
        "model_tokenizer_hash": identity.tokenizer_hash, "files": files,
        "manual_sample": review_keys, "shards": len(chunks), "unique_inputs": len(items),
        "rows_by_task": counts, "test_data_used": False}
    persist(base / "PLAN.json", plan)
    report = {"version": VERSION, "status": "PASS", "phase": "plan", "source_commit": commit,
        "plan_hash": stable_hash(plan), "shards": len(chunks), "unique_inputs": len(items),
        "rows_by_task": counts, "test_data_used": False, "scientific_M2_complete": False,
        "rough_generation_hours_at_engineering_rate": round(len(items) / (96 / (4279.214 + 2043.146) * 3600), 1),
        "engineering_rate_is_not_a_guarantee": True}
    assert_privacy_safe_report(json.dumps(report))
    persist(root / PUBLIC / "plan.json", report)
    # Only this exact v2 identity may seed production. Never import legacy versions.
    env = engineering.environment(root, model_root)
    for job, phase in (("988", "regression"), ("989", "confirmation")):
        for index, original in enumerate(eng_plan["samples"][phase]):
            key = generation.identity(original, config, plan["served_identity"])
            if key not in items:
                continue
            destination = base / "generations" / key
            if (destination / "SEALED.json").exists():
                verified_row(root, plan, key, items[key], env)
                continue
            source = root / engineering.PRIVATE / job / f"item-{index:02d}"
            result = generation.verify_item(source, original, *env, plan["served_identity"], plan["tokenizer_hash"])
            persist(destination / "source.json", items[key])
            for path in source.glob("*.json"):
                if path.name != "source.json":
                    persist(destination / path.name, old.read(path))
            if result["status"] != "PASS":
                raise ContractError("Rejected engineering row cannot seed production.")
            persist(destination / "SEALED.json", {"plan_hash": stable_hash(plan),
                "files": {p.name: file_sha256(p) for p in destination.glob("*.json") if p.name != "SEALED.json"}})
            verified_row(root, plan, key, items[key], env)
    return report


def load_plan(root: Path, commit: str) -> dict:
    plan = old.read(root / PRIVATE / "PLAN.json")
    report = old.read(root / PUBLIC / "plan.json")
    if (plan["execution_commit"] != commit or plan["contract"] != contract(root)
            or report.get("plan_hash") != stable_hash(plan) or report.get("status") != "PASS"
            or plan.get("test_data_used") is not False):
        raise ContractError("Production plan/source/contract mismatch.")
    return plan


def planned_file(root: Path, plan: dict, name: str) -> dict:
    relative = str(PRIVATE / "plan" / name)
    return old.read(checked_file(root, relative, plan["files"][relative]))


def verified_row(root: Path, plan: dict, key: str, item: dict, env: tuple) -> dict:
    config, thresholds, counter = env
    directory = root / PRIVATE / "generations" / key
    seal = old.read(directory / "SEALED.json")
    expected = {p.name: file_sha256(p) for p in directory.glob("*.json") if p.name != "SEALED.json"}
    if seal != {"files": expected, "plan_hash": stable_hash(plan)}:
        raise ContractError("Summary evidence hash or plan binding changed.")
    result = generation.verify_item(directory, item, config, thresholds, counter,
                                    plan["served_identity"], plan["tokenizer_hash"])
    if key != result["generation_key"]:
        raise ContractError("Summary cache key mismatch.")
    return result


def generate_row(root: Path, plan: dict, key: str, item: dict, env: tuple, client=None) -> dict:
    """Resume saved responses without resetting the lifetime two-attempt budget."""
    directory = root / PRIVATE / "generations" / key
    if (directory / "SEALED.json").exists():
        return verified_row(root, plan, key, item, env)
    config, thresholds, counter = env
    if generation.identity(item, config, plan["served_identity"]) != key:
        raise ContractError("Wrong generation request identity.")
    persist(directory / "source.json", item)
    persist(directory / "request.json", generation.payload(item["source"], config))
    client = client or old.PrivateClient(config)
    if (directory / "assessment.json").exists():
        result = old.read(directory / "assessment.json")
        if result["status"] != "PASS":
            return result  # A scientific rejection is not a transient resumable timeout.
    else:
        errors, rejected = [], set()
        result = {"status": "FAIL", "errors": errors}
        for attempt in (1, 2):
            response_path = directory / f"response-{attempt}.json"
            transport_path = directory / f"transport-{attempt}.json"
            started_path = directory / f"started-{attempt}.json"
            try:
                if transport_path.exists():
                    raise RuntimeError("Previously recorded transport failure")
                if not response_path.exists():
                    if started_path.exists():
                        # Interruption during a request: count it, do not silently repeat it.
                        persist(transport_path, {"error": "interrupted_request"})
                        raise RuntimeError("Interrupted request")
                    persist(started_path, {"attempt": attempt})
                    try:
                        response = client.request("chat/completions", generation.payload(item["source"], config))
                    except RuntimeError:
                        persist(transport_path, {"error": "endpoint_failure"})
                        raise
                    persist(response_path, response)
                document = old.response_document(old.read(response_path))
                digest = stable_hash(document)
                if digest in rejected:
                    raise SummaryContractError("identical_rejected_content")
                rejected.add(digest)
                validation = renderer.validate(document, item["source"], thresholds=thresholds,
                                               token_counter=counter)
                text = renderer.render(document)
                result = {"status": "PASS", "document": document, "summary": text,
                    "validation": validation, "attempts": attempt, "errors": errors,
                    "generation_key": key, "embedding_input_key": renderer.embedding_identity(
                        key, text, plan["tokenizer_hash"])}
                break
            except SummaryContractError as exc:
                errors.append(str(exc))
            except old.NonRetryableSummaryError:
                errors.append("embedding_token_budget")
            except RuntimeError:
                errors.append("endpoint_failure")
        persist(directory / "assessment.json", result)
    if result["status"] == "PASS":
        generation.verify_item(directory, item, config, thresholds, counter,
                               plan["served_identity"], plan["tokenizer_hash"])
        persist(directory / "SEALED.json", {"plan_hash": stable_hash(plan),
            "files": {p.name: file_sha256(p) for p in directory.glob("*.json") if p.name != "SEALED.json"}})
    return result


def run_shard(root: Path, model_root: Path, commit: str, index: int, job: str) -> dict:
    plan = load_plan(root, commit)
    if (root / PRIVATE / "HALT.json").exists():
        raise ContractError("Production circuit breaker is set; inspect failure evidence, do not retry.")
    if not 0 <= index < plan["shards"]:
        raise ContractError("Invalid frozen shard index.")
    items = planned_file(root, plan, f"shard-{index}.json")["items"]
    if not 0 < len(items) <= CHUNK_SIZE:
        raise ContractError("Unbounded or empty shard.")
    env = engineering.environment(root, model_root)
    if old.tokenizer_fingerprint(model_root) != plan["tokenizer_hash"]:
        raise ContractError("Frozen tokenizer changed.")
    client = old.PrivateClient(env[0])
    if client.identity() != plan["served_identity"]:
        raise ContractError("Endpoint model identity changed.")
    results = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        entries = list(items.items())
        for offset in range(0, len(entries), 2):
            wave = list(pool.map(lambda pair: generate_row(root, plan, *pair, env), entries[offset:offset + 2]))
            results.extend(wave)
            print(json.dumps({"event": "m2_production_progress", "shard": index,
                "checked": len(results), "planned": len(items),
                "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
            if any(r["status"] != "PASS" for r in wave):
                break  # Keep all completed rows; do not spend the rest of this shard blindly.
    if client.identity() != plan["served_identity"]:
        raise ContractError("Endpoint model identity changed during generation.")
    passed = len(results) == len(items) and all(r["status"] == "PASS" for r in results)
    report = {"version": VERSION, "status": "PASS" if passed else "FAIL", "phase": "shard",
        "job_id": job, "shard": index, "plan_hash": stable_hash(plan),
        "source_commit": commit, "all_inputs_accepted": passed, "test_data_used": False,
        "scientific_M2_complete": False, "runtime_seconds": time.monotonic() - started}
    persist(root / PUBLIC / "attempts" / f"shard-{index}-{job}.json", report)
    if not passed:
        with suppress(FileExistsError):
            old.write_once(root / PRIVATE / "HALT.json", {"reason": "rejected_input", "job_id": job,
                                                        "shard": index, "plan_hash": stable_hash(plan)})
        raise ContractError("Shard rejected an input; preserve evidence and review before further attempts.")
    return report
