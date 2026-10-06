"""Reference coverage, 0.6B embeddings, human gate, and validation readout for M2 v4."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m1.report import fit_condition_scope
from temporal_ehr.final_m23 import selection_v4_production as p
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23.m2 import build_manual_fidelity_packet, validate_manual_fidelity_decisions
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, file_sha256
from temporal_ehr.pilot.evaluation import metric_set
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash


def seal_generation(
    root: Path,
    model_root: Path,
    commit: str,
    array: str,
    readiness_job: str,
) -> dict:
    """Recursively validate every generation before producing the human-review packet."""
    plan = p.load_plan(root, commit)
    readiness = p.verify_readiness(root, model_root, commit, readiness_job)
    env = p.engineering.environment(root, model_root)
    seals: dict[str, str] = {}
    results: dict[str, dict] = {}
    for index in range(plan["shards"]):
        job = f"{array}_{index}"
        old._sentinel(root, job, "COMPLETE", commit)
        report = old.read(root / p.PUBLIC / "attempts" / f"shard-{index}-{job}.json")
        if report.get("plan_hash") != stable_hash(plan) or report.get("status") != "PASS":
            raise ContractError("Selection-v4 shard report does not match the frozen plan.")
        for key, item in p.planned_file(root, plan, f"shard-{index}.json")["items"].items():
            results[key] = p.verified_row(root, plan, key, item, env)
            seals[key] = file_sha256(root / p.PRIVATE / "generations" / key / "SEALED.json")
    if len(results) != plan["unique_inputs"]:
        raise ContractError("Selection-v4 generation coverage is incomplete.")

    records: list[dict] = []
    for task, source in p.source_rows(root):
        for row in source:
            row["generation_key"] = p.generation.identity(
                row.pop("item"), env[0], plan["served_identity"]
            )
        frozen = p.planned_file(root, plan, f"{task}.json")["rows"]
        if source != frozen:
            raise ContractError("Frozen v4 rows do not match the canonical M1 source.")
        records.extend({**row, "summary": results[row["generation_key"]]["summary"]} for row in frozen)

    packet, template = build_manual_fidelity_packet(records, histories_per_task_condition=5)
    sampled = p.balanced_fidelity_sample(records, histories_per_task_condition=5)
    sample_keys = [
        {key: row[key] for key in ("task", "condition", "instance_key", "generation_key")}
        for row in sampled
    ]
    if sample_keys != plan["manual_sample"]:
        raise ContractError("Frozen v4 manual-review sample changed.")
    review = root / p.PRIVATE / "review"
    p.persist(review / "packet.json", packet)
    p.persist(review / "decision_template.json", template)
    if not (review / "decisions.json").exists():
        old.write_once(review / "decisions.json", template)
    receipt = {
        "version": p.VERSION,
        "status": "PASS",
        "plan_hash": stable_hash(plan),
        "readiness_job": readiness_job,
        "readiness_hash": stable_hash(readiness),
        "generation_seals": seals,
        "packet_hash": file_sha256(review / "packet.json"),
        "template_hash": file_sha256(review / "decision_template.json"),
    }
    p.persist(root / p.PRIVATE / "GENERATION_COMPLETE.json", receipt)
    report = {
        "version": p.VERSION,
        "status": "PASS",
        "phase": "coverage",
        "source_commit": commit,
        "plan_hash": stable_hash(plan),
        "unique_inputs": len(results),
        "coverage_seal_hash": stable_hash(receipt),
        "manual_review_histories": 40,
        "extraction": {
            "omitted_series": sum(
                len(indices) for result in results.values() for indices in result["selection"].values()
            ),
            "retained_series": sum(
                len(window["facts"]) for result in results.values() for window in result["document"]["windows"]
            ),
            "interpretation": "bounded extraction; an empty omission plan retains the full grouped source",
        },
        "minimum_retention": {
            name: min(result["validation"]["retention"][name] for result in results.values())
            for name in next(iter(results.values()))["validation"]["retention"]
        },
        "test_data_used": False,
        "scientific_M2_complete": False,
    }
    for name in ("omitted_series", "retained_series"):
        count = report["extraction"][name]
        if 0 < count < 20:
            report["extraction"][name] = "<20"
    assert_privacy_safe_report(json.dumps(report))
    p.persist(root / p.PUBLIC / "coverage.json", report)
    return report


def coverage(root: Path, plan: dict) -> dict:
    seal = old.read(root / p.PRIVATE / "GENERATION_COMPLETE.json")
    report = old.read(root / p.PUBLIC / "coverage.json")
    if (
        seal.get("plan_hash") != stable_hash(plan)
        or seal.get("status") != "PASS"
        or report.get("coverage_seal_hash") != stable_hash(seal)
        or len(seal["generation_seals"]) != plan["unique_inputs"]
    ):
        raise ContractError("Selection-v4 coverage seal is incomplete or changed.")
    readiness_job = str(seal.get("readiness_job", ""))
    old._sentinel(root, readiness_job, "COMPLETE", plan["execution_commit"])
    readiness = old.read(root / p.ENGINEERING_PUBLIC / f"readiness-{readiness_job}.json")
    if stable_hash(readiness) != seal.get("readiness_hash"):
        raise ContractError("Selection-v4 readiness report changed after coverage sealing.")
    for key, digest in seal["generation_seals"].items():
        directory = root / p.PRIVATE / "generations" / key
        p.checked_file(root, str((directory / "SEALED.json").relative_to(root)), digest)
        row_seal = old.read(directory / "SEALED.json")
        expected = {
            "plan_hash": stable_hash(plan),
            "files": {
                path.name: file_sha256(path)
                for path in directory.glob("*.json")
                if path.name != "SEALED.json"
            },
        }
        if row_seal != expected:
            raise ContractError("A validated selection-v4 generation changed.")
    return seal


def embedding_cache(root: Path, plan: dict, task: str, *, read_only: bool = False):
    rows = p.planned_file(root, plan, f"{task}.json")["rows"]
    rows = sorted(
        (row for row in rows if row["condition"] == p.C1),
        key=lambda row: (row["split"], row["instance_key"]),
    )
    if not rows or any(row["split"] not in {"train", "validation"} for row in rows):
        raise ContractError("Invalid v4 reference embedding rows.")
    values = [
        old.read(root / p.PRIVATE / "generations" / row["generation_key"] / "assessment.json")
        for row in rows
    ]
    if any(value.get("status") != "PASS" for value in values):
        raise ContractError("Rejected v4 selections cannot be embedded.")
    row_keys = [
        stable_hash({
            "instance": row["instance_key"],
            "split": row["split"],
            "input": value["embedding_input_key"],
        })
        for row, value in zip(rows, values, strict=True)
    ]
    configuration = stable_hash({
        "plan": stable_hash(plan),
        "task": task,
        "row_keys": row_keys,
        "model": plan["model_artifact_hash"],
        "tokenizer": plan["model_tokenizer_hash"],
    })
    directory = root / p.PRIVATE / "embeddings" / task
    metadata = [
        {
            **{
                key: row[key]
                for key in ("task", "split", "instance_key", "target", "condition", "generation_key")
            },
            "condition_embedding_row_index": index,
            "embedding_row_index": index,
            "embedding_cache_key": row_keys[index],
            "embedding_input_key": values[index]["embedding_input_key"],
        }
        for index, row in enumerate(rows)
    ]
    if read_only:
        if old.read(directory / "metadata.json") != {"rows": metadata}:
            raise ContractError("Selection-v4 embedding metadata changed.")
    else:
        p.persist(directory / "metadata.json", {"rows": metadata})
    cache = ResumableEmbeddingCache(
        directory,
        row_keys=row_keys,
        dimension=1024,
        configuration_hash=configuration,
    )
    return cache, metadata, [value["summary"] for value in values]


def embed(root: Path, model_root: Path, commit: str, task: str, job: str) -> dict:
    if task not in old.TASKS:
        raise ContractError("Invalid selection-v4 task.")
    plan = p.load_plan(root, commit)
    coverage(root, plan)
    identity = load_model_identity(
        model_root / "Qwen3-Embedding-0.6B",
        expected_repository="Qwen/Qwen3-Embedding-0.6B",
    )
    if (
        identity.artifact_hash != plan["model_artifact_hash"]
        or identity.tokenizer_hash != plan["model_tokenizer_hash"]
    ):
        raise ContractError("Frozen selection-v4 encoder identity changed.")
    cache, metadata, texts = embedding_cache(root, plan, task)
    if not cache.complete_path.exists():
        pending = cache.pending_indices()
        cache.begin_attempt()
        if len(pending):
            encoder = LocalQwenEncoder(identity, max_seq_length=8192)
            if encoder.dimension != 1024:
                raise ContractError("Wrong selection-v4 encoder dimensionality.")
            for start in range(0, len(pending), 8):
                indices = pending[start : start + 8].tolist()
                cache.write_rows(indices, encoder.encode([texts[i] for i in indices], batch_size=8))
                print(json.dumps({
                    "event": "m2_selection_embedding_progress",
                    "task": task,
                    "completed_this_attempt": start + len(indices),
                    "pending_at_start": len(pending),
                }), flush=True)
        cache.finalize()
        cache.finish_attempt()
    vectors = cache.load_complete()
    if not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1, atol=1e-6, rtol=0):
        raise ContractError("Selection-v4 embedding normalization failed.")
    report = {
        "version": p.VERSION,
        "status": "PASS",
        "task": task,
        "phase": "embed",
        "source_commit": commit,
        "plan_hash": stable_hash(plan),
        "rows": len(metadata),
        "dimension": 1024,
        "metadata_hash": file_sha256(cache.root / "metadata.json"),
        "cache_complete_hash": file_sha256(cache.complete_path),
        "test_data_used": False,
        "scientific_M2_complete": False,
    }
    p.persist(root / p.PUBLIC / "embedding" / f"{task}.json", report)
    p.persist(root / p.PUBLIC / "attempts" / f"embed-{job}.json", {**report, "job_id": job})
    return report


def human_gate(root: Path, plan: dict, *, read_only: bool = False) -> dict:
    seal = coverage(root, plan)
    review = root / p.PRIVATE / "review"
    if file_sha256(review / "packet.json") != seal["packet_hash"]:
        raise ContractError("Selection-v4 review packet changed.")
    decision = old.read(review / "decisions.json")
    result = validate_manual_fidelity_decisions(old.read(review / "packet.json"), decision)
    if result["status"] != "PASS":
        raise ContractError("Manual fidelity review failed; M2 fitting remains blocked.")
    receipt = {
        "packet_hash": seal["packet_hash"],
        "decisions_hash": stable_hash(decision),
        "result": result,
    }
    if read_only:
        if old.read(review / "ACCEPTED.json") != receipt:
            raise ContractError("Selection-v4 manual-review acceptance changed.")
    else:
        p.persist(review / "ACCEPTED.json", receipt)
    return receipt


def fit(root: Path, model_root: Path, commit: str, embed_array: str) -> dict:
    plan = p.load_plan(root, commit)
    manual = human_gate(root, plan)
    spec = old.load_final_m23_authorization(root / old.BASELINE, repository_root=root)
    task_reports: dict[str, dict] = {}
    metrics: list[dict] = []
    for index, task in enumerate(old.TASKS):
        old._sentinel(root, f"{embed_array}_{index}", "COMPLETE", commit)
        cache, metadata, _ = embedding_cache(root, plan, task)
        report = old.read(root / p.PUBLIC / "embedding" / f"{task}.json")
        if (
            report.get("plan_hash") != stable_hash(plan)
            or report.get("status") != "PASS"
            or report.get("metadata_hash") != file_sha256(cache.root / "metadata.json")
            or report.get("cache_complete_hash") != file_sha256(cache.complete_path)
        ):
            raise ContractError("Selection-v4 embedding provenance changed.")
        vectors = cache.load_complete()
        target = root / p.PRIVATE / "fit" / f"{task}.json"
        if target.exists():
            result = old.read(target)
            if result["plan_hash"] != stable_hash(plan) or result["manual_hash"] != stable_hash(manual):
                raise ContractError("Selection-v4 partial fit has different provenance.")
        else:
            task_report, predictions = fit_condition_scope(
                task=task,
                scope="m2_selection_v4_reference",
                condition_data={p.C1: (metadata, vectors)},
                c_grid=spec.evaluation["regularization_grid"],
                random_seed=int(spec.evaluation["logistic_seed"]),
                model_role="M2",
            )
            result = {
                "plan_hash": stable_hash(plan),
                "manual_hash": stable_hash(manual),
                "report": task_report,
                "predictions": predictions,
            }
            p.persist(target, result)
        task_reports[task] = result["report"]
        metrics.append(result["report"]["representations"][p.C1]["metrics"])
    result = {
        "version": p.VERSION,
        "status": "PASS",
        "phase": "fit",
        "source_commit": commit,
        "plan_hash": stable_hash(plan),
        "tasks": task_reports,
        "macro_metrics": {key: float(np.mean([value[key] for value in metrics])) for key in metrics[0]},
        "manual_review": manual["result"],
        "manual_receipt_hash": stable_hash(manual),
        "protocol_decision": "M2_SELECTION_V3_REFERENCE_VALIDATION_COMPLETE_TEST_REMAINS_CLOSED",
        "reference_complete": True,
        "sensitivity_complete": False,
        "test_data_used": False,
        "scientific_M2_complete": False,
        "interpretation": "Post-M1 exploratory validation reference; no new held-out selection.",
        "private_fit_hashes": {
            task: file_sha256(root / p.PRIVATE / "fit" / f"{task}.json")
            for task in old.TASKS
        },
    }
    assert_privacy_safe_report(json.dumps(result))
    p.persist(root / p.PUBLIC / "reference_validation.json", result)
    return result


def verify_reference(root: Path, commit: str, embed_array: str, fit_job: str) -> dict:
    """Read-only final verifier: never generate, encode, fit, or repair."""
    old._sentinel(root, fit_job, "COMPLETE", commit)
    plan = p.load_plan(root, commit)
    manual = human_gate(root, plan, read_only=True)
    report = old.read(root / p.PUBLIC / "reference_validation.json")
    if (
        report.get("status") != "PASS"
        or report.get("plan_hash") != stable_hash(plan)
        or report.get("source_commit") != commit
        or report.get("test_data_used") is not False
        or report.get("manual_receipt_hash") != stable_hash(manual)
        or report.get("protocol_decision")
        != "M2_SELECTION_V3_REFERENCE_VALIDATION_COMPLETE_TEST_REMAINS_CLOSED"
        or set(report.get("tasks", {})) != set(old.TASKS)
    ):
        raise ContractError("Incomplete or stale selection-v4 M2 aggregate.")
    metrics: list[dict] = []
    for index, task in enumerate(old.TASKS):
        old._sentinel(root, f"{embed_array}_{index}", "COMPLETE", commit)
        cache, metadata, _ = embedding_cache(root, plan, task, read_only=True)
        cache.load_complete()
        path = root / p.PRIVATE / "fit" / f"{task}.json"
        private = old.read(path)
        if file_sha256(path) != report["private_fit_hashes"][task] or private["report"] != report["tasks"][task]:
            raise ContractError("Selection-v4 private fit/report hash mismatch.")
        predictions = private["predictions"]
        expected = {row["instance_key"]: row["target"] for row in metadata if row["split"] == "validation"}
        if (
            len(predictions) != len(expected)
            or {row["instance_key"]: row["target"] for row in predictions} != expected
            or any(row["task"] != task or row["model_role"] != "M2" for row in predictions)
        ):
            raise ContractError("Selection-v4 validation predictions do not cover the frozen set.")
        value = metric_set(
            np.asarray([row["target"] for row in predictions]),
            np.asarray([row["probability"] for row in predictions]),
        )
        if value != report["tasks"][task]["representations"][p.C1]["metrics"]:
            raise ContractError("Selection-v4 reported metrics changed.")
        metrics.append(value)
    expected_macro = {key: float(np.mean([value[key] for value in metrics])) for key in metrics[0]}
    if report["macro_metrics"] != expected_macro:
        raise ContractError("Selection-v4 macro metrics changed.")
    return {
        "version": p.VERSION,
        "status": "PASS",
        "phase": "verify",
        "reference_complete": True,
        "sensitivity_complete": False,
        "test_data_used": False,
        "reference_report_hash": file_sha256(root / p.PUBLIC / "reference_validation.json"),
    }
