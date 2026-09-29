"""Hash-bound reference coverage, 0.6B embeddings, human gate, validation readout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m1.report import fit_condition_scope
from temporal_ehr.final_m23 import structured_production as p
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_v2_recovery as engineering
from temporal_ehr.final_m23.m2 import build_manual_fidelity_packet, validate_manual_fidelity_decisions
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, file_sha256
from temporal_ehr.pilot.evaluation import metric_set
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash


def seal_generation(root: Path, model_root: Path, commit: str, array: str) -> dict:
    plan = p.load_plan(root, commit)
    env = engineering.environment(root, model_root)
    seals, results = {}, {}
    for index in range(plan["shards"]):
        job = f"{array}_{index}"
        old._sentinel(root, job, "COMPLETE", commit)
        report = old.read(root / p.PUBLIC / "attempts" / f"shard-{index}-{job}.json")
        if report.get("plan_hash") != stable_hash(plan) or report.get("status") != "PASS":
            raise ContractError("Generation shard report does not match frozen plan.")
        for key, item in p.planned_file(root, plan, f"shard-{index}.json")["items"].items():
            results[key] = p.verified_row(root, plan, key, item, env)
            seals[key] = file_sha256(root / p.PRIVATE / "generations" / key / "SEALED.json")
    if len(results) != plan["unique_inputs"]:
        raise ContractError("Generation coverage incomplete.")
    # Reconstruct source selection; never trust labels or patient mappings from a new file alone.
    records = []
    for task, source in p.source_rows(root):
        for row in source:
            row["generation_key"] = p.generation.identity(row.pop("item"), env[0], plan["served_identity"])
        frozen = p.planned_file(root, plan, f"{task}.json")["rows"]
        if source != frozen:
            raise ContractError("Frozen production rows do not match canonical M1 source.")
        records.extend({**r, "summary": results[r["generation_key"]]["summary"]} for r in frozen)
    packet, template = build_manual_fidelity_packet(records, histories_per_task_condition=5)
    sampled = p.balanced_fidelity_sample(records, histories_per_task_condition=5)
    sample_keys = [{k: r[k] for k in ("task", "condition", "instance_key", "generation_key")} for r in sampled]
    if sample_keys != plan["manual_sample"]:
        raise ContractError("Frozen manual sample changed.")
    review_dir = root / p.PRIVATE / "review"
    p.persist(review_dir / "packet.json", packet)
    p.persist(review_dir / "decision_template.json", template)
    # Writable working copy is intentionally outside the immutable packet seal.
    if not (review_dir / "decisions.json").exists():
        old.write_once(review_dir / "decisions.json", template)
    receipt = {"version": p.VERSION, "status": "PASS", "plan_hash": stable_hash(plan),
               "generation_seals": seals, "packet_hash": file_sha256(review_dir / "packet.json"),
               "template_hash": file_sha256(review_dir / "decision_template.json")}
    p.persist(root / p.PRIVATE / "GENERATION_COMPLETE.json", receipt)
    report = {"version": p.VERSION, "status": "PASS", "phase": "coverage",
        "source_commit": commit, "plan_hash": stable_hash(plan), "unique_inputs": len(results),
        "coverage_seal_hash": stable_hash(receipt), "manual_review_histories": 40,
        "minimum_retention": {k: min(r["validation"]["retention"][k] for r in results.values())
                              for k in next(iter(results.values()))["validation"]["retention"]},
        "test_data_used": False, "scientific_M2_complete": False}
    assert_privacy_safe_report(json.dumps(report))
    p.persist(root / p.PUBLIC / "coverage.json", report)
    return report


def coverage(root: Path, plan: dict) -> dict:
    seal = old.read(root / p.PRIVATE / "GENERATION_COMPLETE.json")
    report = old.read(root / p.PUBLIC / "coverage.json")
    if (seal.get("plan_hash") != stable_hash(plan) or seal.get("status") != "PASS"
            or report.get("coverage_seal_hash") != stable_hash(seal)
            or len(seal["generation_seals"]) != plan["unique_inputs"]):
        raise ContractError("Generation coverage seal is incomplete or changed.")
    for key, digest in seal["generation_seals"].items():
        directory = root / p.PRIVATE / "generations" / key
        p.checked_file(root, str((directory / "SEALED.json").relative_to(root)), digest)
        row_seal = old.read(directory / "SEALED.json")
        if row_seal != {"plan_hash": stable_hash(plan), "files": {
                path.name: file_sha256(path) for path in directory.glob("*.json") if path.name != "SEALED.json"}}:
            raise ContractError("Previously validated generation bytes changed.")
    return seal


def embedding_cache(root: Path, plan: dict, task: str, *, read_only: bool = False):
    rows = p.planned_file(root, plan, f"{task}.json")["rows"]
    rows = sorted((r for r in rows if r["condition"] == p.C1), key=lambda r: (r["split"], r["instance_key"]))
    if not rows or any(r["split"] not in {"train", "validation"} for r in rows):
        raise ContractError("Invalid reference embedding rows.")
    values = [old.read(root / p.PRIVATE / "generations" / r["generation_key"] / "assessment.json") for r in rows]
    if any(v.get("status") != "PASS" for v in values):
        raise ContractError("Rejected summaries cannot be embedded.")
    row_keys = [stable_hash({"instance": r["instance_key"], "split": r["split"], "input": v["embedding_input_key"]})
                for r, v in zip(rows, values, strict=True)]
    conf = stable_hash({"plan": stable_hash(plan), "task": task, "row_keys": row_keys,
                       "model": plan["model_artifact_hash"], "tokenizer": plan["model_tokenizer_hash"]})
    directory = root / p.PRIVATE / "embeddings" / task
    metadata = [{**{k: r[k] for k in ("task", "split", "instance_key", "target", "condition", "generation_key")},
                 "condition_embedding_row_index": i, "embedding_row_index": i,
                 "embedding_cache_key": row_keys[i], "embedding_input_key": values[i]["embedding_input_key"]}
                for i, r in enumerate(rows)]
    if read_only:
        if old.read(directory / "metadata.json") != {"rows": metadata}:
            raise ContractError("Embedding row metadata changed.")
    else:
        p.persist(directory / "metadata.json", {"rows": metadata})
    cache = ResumableEmbeddingCache(directory, row_keys=row_keys, dimension=1024, configuration_hash=conf)
    return cache, metadata, [v["summary"] for v in values]


def embed(root: Path, model_root: Path, commit: str, task: str, job: str) -> dict:
    if task not in old.TASKS:
        raise ContractError("Invalid task.")
    plan = p.load_plan(root, commit)
    coverage(root, plan)
    identity = load_model_identity(model_root / "Qwen3-Embedding-0.6B",
                                   expected_repository="Qwen/Qwen3-Embedding-0.6B")
    if identity.artifact_hash != plan["model_artifact_hash"] or identity.tokenizer_hash != plan["model_tokenizer_hash"]:
        raise ContractError("Frozen encoder identity changed.")
    cache, metadata, texts = embedding_cache(root, plan, task)
    if not cache.complete_path.exists():
        pending = cache.pending_indices()
        cache.begin_attempt()
        if len(pending):
            encoder = LocalQwenEncoder(identity, max_seq_length=8192)
            if encoder.dimension != 1024:
                raise ContractError("Wrong encoder dimensionality.")
            for start in range(0, len(pending), 8):
                indices = pending[start:start + 8].tolist()
                cache.write_rows(indices, encoder.encode([texts[i] for i in indices], batch_size=8))
                print(json.dumps({"event": "m2_embedding_progress", "task": task,
                                  "completed_this_attempt": start + len(indices),
                                  "pending_at_start": len(pending)}), flush=True)
        cache.finalize()
        cache.finish_attempt()
    vectors = cache.load_complete()
    if not np.allclose(np.linalg.norm(vectors.astype(np.float64), axis=1), 1, atol=1e-6, rtol=0):
        raise ContractError("Embedding normalization failed.")
    report = {"version": p.VERSION, "status": "PASS", "task": task, "phase": "embed",
        "source_commit": commit, "plan_hash": stable_hash(plan), "rows": len(metadata), "dimension": 1024,
        "metadata_hash": file_sha256(cache.root / "metadata.json"),
        "cache_complete_hash": file_sha256(cache.complete_path), "test_data_used": False,
        "scientific_M2_complete": False}
    p.persist(root / p.PUBLIC / "embedding" / f"{task}.json", report)
    p.persist(root / p.PUBLIC / "attempts" / f"embed-{job}.json", {**report, "job_id": job})
    return report


def human_gate(root: Path, plan: dict, *, read_only: bool = False) -> dict:
    seal = coverage(root, plan)
    review = root / p.PRIVATE / "review"
    if file_sha256(review / "packet.json") != seal["packet_hash"]:
        raise ContractError("Frozen manual review packet changed.")
    decision = old.read(review / "decisions.json")
    result = validate_manual_fidelity_decisions(old.read(review / "packet.json"), decision)
    if result["status"] != "PASS":
        raise ContractError("Manual fidelity review failed; M2 is not eligible for fitting.")
    # Immutable acceptance record makes later edits to working decisions detectable.
    receipt = {"packet_hash": seal["packet_hash"], "decisions_hash": stable_hash(decision), "result": result}
    if read_only:
        if old.read(review / "ACCEPTED.json") != receipt:
            raise ContractError("Manual review acceptance changed.")
    else:
        p.persist(review / "ACCEPTED.json", receipt)
    return receipt


def fit(root: Path, model_root: Path, commit: str, embed_array: str) -> dict:
    plan = p.load_plan(root, commit)
    manual = human_gate(root, plan)  # Before labels are consumed by any estimator.
    spec = old.load_final_m23_authorization(root / old.BASELINE, repository_root=root)
    task_reports, metrics = {}, []
    for index, task in enumerate(old.TASKS):
        old._sentinel(root, f"{embed_array}_{index}", "COMPLETE", commit)
        cache, metadata, _ = embedding_cache(root, plan, task)
        report = old.read(root / p.PUBLIC / "embedding" / f"{task}.json")
        if (report.get("plan_hash") != stable_hash(plan) or report.get("status") != "PASS"
                or report.get("metadata_hash") != file_sha256(cache.root / "metadata.json")
                or report.get("cache_complete_hash") != file_sha256(cache.complete_path)):
            raise ContractError("Embedding report identity or hashes changed.")
        vectors = cache.load_complete()
        target = root / p.PRIVATE / "fit" / f"{task}.json"
        if target.exists():
            result = old.read(target)
            if result["plan_hash"] != stable_hash(plan) or result["manual_hash"] != stable_hash(manual):
                raise ContractError("Partial fit belongs to different provenance.")
        else:
            task_report, predictions = fit_condition_scope(task=task, scope="m2_reference",
                condition_data={p.C1: (metadata, vectors)},
                c_grid=spec.evaluation["regularization_grid"], random_seed=int(spec.evaluation["logistic_seed"]),
                model_role="M2")
            result = {"plan_hash": stable_hash(plan), "manual_hash": stable_hash(manual),
                      "report": task_report, "predictions": predictions}
            p.persist(target, result)
        task_reports[task] = result["report"]
        metrics.append(result["report"]["representations"][p.C1]["metrics"])
    result = {"version": p.VERSION, "status": "PASS", "phase": "fit",
        "source_commit": commit, "plan_hash": stable_hash(plan), "tasks": task_reports,
        "macro_metrics": {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
        "manual_review": manual["result"], "manual_receipt_hash": stable_hash(manual),
        "protocol_decision": "M2_REFERENCE_VALIDATION_COMPLETE_TEST_REMAINS_CLOSED",
        "reference_complete": True, "sensitivity_complete": False, "test_data_used": False,
        "scientific_M2_complete": False,
        "interpretation": "Post-M1 exploratory validation reference; no new held-out selection.",
        "private_fit_hashes": {t: file_sha256(root / p.PRIVATE / "fit" / f"{t}.json") for t in old.TASKS}}
    assert_privacy_safe_report(json.dumps(result))
    p.persist(root / p.PUBLIC / "reference_validation.json", result)
    return result


def verify_reference(root: Path, commit: str, embed_array: str, fit_job: str) -> dict:
    """Read-only final verifier: never fit, generate, encode, or repair missing files."""
    old._sentinel(root, fit_job, "COMPLETE", commit)
    plan = p.load_plan(root, commit)
    manual = human_gate(root, plan, read_only=True)
    report = old.read(root / p.PUBLIC / "reference_validation.json")
    if (report.get("status") != "PASS" or report.get("plan_hash") != stable_hash(plan)
            or report.get("source_commit") != commit or report.get("test_data_used") is not False
            or report.get("manual_receipt_hash") != stable_hash(manual)
            or report.get("protocol_decision") != "M2_REFERENCE_VALIDATION_COMPLETE_TEST_REMAINS_CLOSED"
            or set(report.get("tasks", {})) != set(old.TASKS)):
        raise ContractError("Incomplete or stale M2 reference aggregate.")
    metrics = []
    for index, task in enumerate(old.TASKS):
        old._sentinel(root, f"{embed_array}_{index}", "COMPLETE", commit)
        cache, metadata, _ = embedding_cache(root, plan, task, read_only=True)
        cache.load_complete()
        path = root / p.PRIVATE / "fit" / f"{task}.json"
        if file_sha256(path) != report["private_fit_hashes"][task] or old.read(path)["report"] != report["tasks"][task]:
            raise ContractError("Private readout/result hash mismatch.")
        predictions = old.read(path)["predictions"]
        expected = {row["instance_key"]: row["target"] for row in metadata if row["split"] == "validation"}
        if (len(predictions) != len(expected)
                or {row["instance_key"]: row["target"] for row in predictions} != expected
                or any(row["task"] != task or row["model_role"] != "M2" for row in predictions)):
            raise ContractError("Validation predictions do not cover the frozen patient set.")
        value = metric_set(np.asarray([row["target"] for row in predictions]),
                           np.asarray([row["probability"] for row in predictions]))
        if value != report["tasks"][task]["representations"][p.C1]["metrics"]:
            raise ContractError("Reported metrics do not match validation predictions.")
        metrics.append(value)
    if report["macro_metrics"] != {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}:
        raise ContractError("Equal-weight macro metrics changed.")
    return {"version": p.VERSION, "status": "PASS", "phase": "verify",
            "reference_complete": True, "sensitivity_complete": False, "test_data_used": False,
            "reference_report_hash": file_sha256(root / p.PUBLIC / "reference_validation.json")}
