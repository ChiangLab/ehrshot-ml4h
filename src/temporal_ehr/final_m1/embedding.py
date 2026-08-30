"""Condition-level resumable embedding caches for final primary and confirmatory M1."""

from __future__ import annotations

import gc
import json
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import FinalM1Authorization, load_final_m1_authorization
from temporal_ehr.final_m1.foundation import _write_immutable_json
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, embedding_cache_key, file_sha256
from temporal_ehr.pilot.model import LocalQwenEncoder, ModelIdentity, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash


def condition_label(row: Mapping[str, Any]) -> str:
    seed = row.get("perturbation_seed")
    return str(row["representation"]) if seed is None else f"{row['representation']}:seed={seed}"


def condition_slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def _model_slug(role: str, identity: ModelIdentity) -> str:
    size = identity.repository.rsplit("-", 1)[-1].casefold()
    return f"{role}_{size}"


def _condition_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[condition_label(row)].append(dict(row))
    for condition_rows in grouped.values():
        condition_rows.sort(key=lambda row: (str(row["split"]), str(row["instance_key"])))
    return dict(sorted(grouped.items()))


def _cache_row_key(
    row: Mapping[str, Any],
    *,
    spec: FinalM1Authorization,
    identity: ModelIdentity,
) -> str:
    return embedding_cache_key(
        experiment_id=spec.experiment_id,
        task=str(row["task"]),
        instance_key=str(row["instance_key"]),
        representation=str(row["representation"]),
        perturbation_seed=row.get("perturbation_seed"),
        text_hash=str(row["text_hash"]),
        model_hash=identity.artifact_hash,
        tokenizer_hash=identity.tokenizer_hash,
        instruction_hash=stable_hash(spec.instructions[str(row["task"])].strip()),
        budget=spec.token_budget,
    )


def _metadata_table(
    rows: Sequence[Mapping[str, Any]],
    row_keys: Sequence[str],
    *,
    identity: ModelIdentity,
    role: str,
) -> pa.Table:
    metadata = []
    for index, (row, row_key) in enumerate(zip(rows, row_keys, strict=True)):
        metadata.append(
            {
                "experiment_id": row["experiment_id"],
                "task": row["task"],
                "split": row["split"],
                "instance_key": row["instance_key"],
                "subject_id": row["subject_id"],
                "prediction_time": row["prediction_time"],
                "target": row["target"],
                "representation": row["representation"],
                "perturbation_seed": row.get("perturbation_seed"),
                "condition_embedding_row_index": index,
                "condition_cache_key": row_key,
                "text_hash": row["text_hash"],
                "model_role": role,
                "model_repository": identity.repository,
                "model_artifact_hash": identity.artifact_hash,
                "tokenizer_hash": identity.tokenizer_hash,
                "mirrors_embedding_row": True,
            }
        )
    return pa.Table.from_pylist(metadata)


def _write_or_verify_metadata(path: Path, table: pa.Table) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        pq.write_table(table, path, compression="zstd")
    else:
        existing = pq.read_table(path)
        if not existing.equals(table):
            raise ContractError("Existing condition metadata does not match the immutable embedding row order.")
    return file_sha256(path)


def _embed_condition(
    *,
    spec: FinalM1Authorization,
    task: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    role: str,
    identity: ModelIdentity,
    encoder: LocalQwenEncoder,
    batch_size: int,
    source_git_commit: str,
    prepared_texts_sha256: str,
) -> dict[str, Any]:
    row_keys = [_cache_row_key(row, spec=spec, identity=identity) for row in rows]
    if len(row_keys) != len(set(row_keys)):
        raise ContractError("One final condition contains duplicate model-specific cache keys.")
    if role == "primary" and row_keys != [str(row["cache_key"]) for row in rows]:
        raise ContractError("Primary final cache keys drifted from the prepared patient-window metadata.")
    root = spec.private_root / "embeddings" / _model_slug(role, identity) / task / condition_slug(label)
    configuration_hash = stable_hash(
        {
            "namespace": "final_condition_cache_v1",
            "experiment_id": spec.experiment_id,
            "experiment_config_hash": stable_hash(spec.raw),
            "source_git_commit": source_git_commit,
            "prepared_texts_sha256": prepared_texts_sha256,
            "task": task,
            "condition": label,
            "model_repository": identity.repository,
            "model_artifact_hash": identity.artifact_hash,
            "tokenizer_hash": identity.tokenizer_hash,
            "token_budget": spec.token_budget,
            "row_keys_hash": stable_hash(row_keys),
        }
    )
    metadata_hash = _write_or_verify_metadata(
        root / "patient_window_metadata.parquet",
        _metadata_table(rows, row_keys, identity=identity, role=role),
    )
    cache = ResumableEmbeddingCache(
        root,
        row_keys=row_keys,
        dimension=encoder.dimension,
        configuration_hash=configuration_hash,
    )
    if cache.complete_path.is_file():
        vectors = cache.load_complete()
        norms = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6, rtol=0.0):
            raise ContractError("Verified final cache contains non-unit embeddings.")
        summary_path = root / "SUMMARY.json"
        if not summary_path.is_file():
            attempts = (
                json.loads(cache.attempts_path.read_text(encoding="utf-8"))
                if cache.attempts_path.is_file()
                else {}
            )
            if attempts.get("attempt_active") is True:
                attempts = cache.finish_attempt()
            recovered = {
                "status": "PASS",
                "experiment_id": spec.experiment_id,
                "source_git_commit": source_git_commit,
                "task": task,
                "condition": label,
                "model_role": role,
                "model_repository": identity.repository,
                "model_revision_hash": stable_hash(identity.revision),
                "model_artifact_hash": identity.artifact_hash,
                "tokenizer_hash": identity.tokenizer_hash,
                "configuration_hash": configuration_hash,
                "rows": len(rows),
                "dimension": encoder.dimension,
                "full_embedding_dimension_retained": True,
                "l2_normalized_in_float32": True,
                "metadata_rows": len(rows),
                "metadata_sha256": metadata_hash,
                "metadata_mirrors_every_embedding_row": True,
                "rows_computed_this_attempt": 0,
                "cache_reused_without_encoding": True,
                "attempt_number": int(attempts.get("attempt_count", 0)),
                "failures": int(attempts.get("failed_attempts", 0)),
                "retries": int(attempts.get("retries", 0)),
                "elapsed_seconds": 0.0,
                "documents_per_second": None,
                "peak_gpu_memory_bytes": 0,
                "cache_complete": True,
                "cache_complete_sha256": file_sha256(cache.complete_path),
                "test_split_used": False,
                "summary_recovered_from_verified_complete_cache": True,
            }
            _write_immutable_json(summary_path, recovered)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("configuration_hash") != configuration_hash
            or summary.get("metadata_sha256") != metadata_hash
            or summary.get("cache_complete_sha256") != file_sha256(cache.complete_path)
        ):
            raise ContractError("Completed final condition cache summary drifted.")
        reused = dict(summary)
        reused["cache_reused_without_encoding"] = True
        reused["rows_computed_this_attempt"] = 0
        return reused

    attempt = cache.begin_attempt()
    pending = cache.pending_indices()
    encoder.reset_peak_memory()
    started = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        indices = pending[offset : offset + batch_size]
        vectors = encoder.encode([str(rows[int(index)]["text"]) for index in indices], batch_size=batch_size)
        cache.write_rows(indices.tolist(), vectors)
    cache.finalize()
    attempt = cache.finish_attempt()
    elapsed = time.perf_counter() - started
    embeddings = cache.load_complete()
    norms = np.linalg.norm(np.asarray(embeddings, dtype=np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0.0):
        raise ContractError("Final condition embeddings failed full-cache FP32 unit-norm validation.")
    summary = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "task": task,
        "condition": label,
        "model_role": role,
        "model_repository": identity.repository,
        "model_revision_hash": stable_hash(identity.revision),
        "model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "configuration_hash": configuration_hash,
        "rows": len(rows),
        "dimension": encoder.dimension,
        "full_embedding_dimension_retained": True,
        "l2_normalized_in_float32": True,
        "metadata_rows": len(rows),
        "metadata_sha256": metadata_hash,
        "metadata_mirrors_every_embedding_row": True,
        "rows_computed_this_attempt": len(pending),
        "cache_reused_without_encoding": False,
        "attempt_number": attempt["attempt_count"],
        "failures": attempt["failed_attempts"],
        "retries": attempt["retries"],
        "elapsed_seconds": elapsed,
        "documents_per_second": len(pending) / elapsed if elapsed and len(pending) else None,
        "peak_gpu_memory_bytes": encoder.peak_memory_bytes(),
        "cache_complete": True,
        "cache_complete_sha256": file_sha256(cache.complete_path),
        "test_split_used": False,
    }
    _write_immutable_json(root / "SUMMARY.json", summary)
    return summary


def _selected_confirmatory_priorities(
    throughput: Mapping[str, Any],
    task: str,
) -> tuple[str, ...]:
    try:
        selected = throughput["feasibility_plan"]["tasks"][task]["confirmatory_priorities_selected"]
    except (KeyError, TypeError) as exc:
        raise ContractError("Throughput report lacks a task-specific 8B feasibility decision.") from exc
    if not isinstance(selected, list):
        raise ContractError("8B feasibility selection is not an ordered list.")
    canonical = list(throughput["feasibility_plan"]["confirmatory_priority"])
    if selected != canonical[: len(selected)]:
        raise ContractError("8B confirmatory selection violated its prespecified priority order.")
    return tuple(str(value) for value in selected)


def embed_final_task(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    task: str,
    source_git_commit: str,
) -> dict[str, Any]:
    """Embed every primary condition and only runtime-authorized 8B confirmatory cells."""
    spec = load_final_m1_authorization(config_path, repository_root=repository_root)
    if task not in spec.tasks:
        raise ContractError(f"Task {task} is not in the locked final panel.")
    throughput_path = spec.aggregate_root / "throughput.json"
    if not throughput_path.is_file():
        raise ContractError("Final embedding requires a completed throughput gate.")
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    if (
        throughput.get("status") != "PASS"
        or throughput.get("source_git_commit") != source_git_commit
        or throughput.get("outcomes_or_performance_read") is not False
        or throughput.get("test_predictions_or_performance_used") is not False
    ):
        raise ContractError("Final throughput identity or no-outcome gate failed.")
    rows = read_final_task_texts(spec, task)
    grouped = _condition_rows(rows)
    texts_sha256 = file_sha256(spec.private_root / "prepared" / task / "texts.parquet")
    primary_identity = load_model_identity(
        model_root / spec.primary_model_directory_name,
        expected_repository=spec.primary_model_repository,
    )
    primary_encoder = LocalQwenEncoder(primary_identity, max_seq_length=spec.token_budget)
    primary_summaries = [
        _embed_condition(
            spec=spec,
            task=task,
            label=label,
            rows=condition_rows,
            role="primary",
            identity=primary_identity,
            encoder=primary_encoder,
            batch_size=int(spec.raw["models"]["primary"]["batch_size"]),
            source_git_commit=source_git_commit,
            prepared_texts_sha256=texts_sha256,
        )
        for label, condition_rows in grouped.items()
    ]
    del primary_encoder
    gc.collect()
    selected_priorities = _selected_confirmatory_priorities(throughput, task)
    confirmatory_summaries: list[dict[str, Any]] = []
    if selected_priorities:
        confirmatory_identity = load_model_identity(
            model_root / spec.confirmatory_model_directory_name,
            expected_repository=spec.confirmatory_model_repository,
        )
        confirmatory_encoder = LocalQwenEncoder(confirmatory_identity, max_seq_length=spec.token_budget)
        for label, condition_rows in grouped.items():
            representation = str(condition_rows[0]["representation"])
            if representation not in selected_priorities:
                continue
            confirmatory_summaries.append(
                _embed_condition(
                    spec=spec,
                    task=task,
                    label=label,
                    rows=condition_rows,
                    role="confirmatory",
                    identity=confirmatory_identity,
                    encoder=confirmatory_encoder,
                    batch_size=int(spec.raw["models"]["confirmatory"]["batch_size"]),
                    source_git_commit=source_git_commit,
                    prepared_texts_sha256=texts_sha256,
                )
            )
        del confirmatory_encoder
        gc.collect()
    report = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "source_git_commit": source_git_commit,
        "task": task,
        "primary_model_repository": primary_identity.repository,
        "primary_conditions": primary_summaries,
        "primary_condition_count": len(grouped),
        "confirmatory_priorities_selected_by_runtime": list(selected_priorities),
        "confirmatory_conditions": confirmatory_summaries,
        "condition_level_caches": True,
        "immutable_verified_cache_reuse": True,
        "pilot_cache_reused": False,
        "metadata_mirrors_every_embedding_row": True,
        "test_split_used": False,
    }
    _write_immutable_json(spec.aggregate_root / "embedding" / f"{task}.json", report, aggregate=True)
    return report


def load_final_condition_embeddings(
    *,
    spec: FinalM1Authorization,
    task: str,
    label: str,
    role: str,
    identity: ModelIdentity,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Hash-verify one complete cache and its one-to-one private metadata table."""
    root = spec.private_root / "embeddings" / _model_slug(role, identity) / task / condition_slug(label)
    summary_path = root / "SUMMARY.json"
    cache_metadata_path = root / "CACHE.json"
    metadata_path = root / "patient_window_metadata.parquet"
    if not summary_path.is_file() or not cache_metadata_path.is_file() or not metadata_path.is_file():
        raise ContractError(f"Final condition cache is incomplete: {role}/{task}/{label}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    metadata = pq.read_table(metadata_path).to_pylist()
    metadata.sort(key=lambda row: int(row["condition_embedding_row_index"]))
    if (
        summary.get("status") != "PASS"
        or summary.get("experiment_id") != spec.experiment_id
        or summary.get("task") != task
        or summary.get("condition") != label
        or summary.get("model_role") != role
        or summary.get("model_repository") != identity.repository
        or summary.get("model_artifact_hash") != identity.artifact_hash
        or summary.get("tokenizer_hash") != identity.tokenizer_hash
        or summary.get("configuration_hash") != cache_metadata.get("configuration_hash")
    ):
        raise ContractError("Final condition summary/model/configuration identity drifted.")
    if any(
        row.get("experiment_id") != spec.experiment_id
        or row.get("task") != task
        or condition_label(row) != label
        or row.get("model_role") != role
        or row.get("model_repository") != identity.repository
        or row.get("model_artifact_hash") != identity.artifact_hash
        or row.get("tokenizer_hash") != identity.tokenizer_hash
        for row in metadata
    ):
        raise ContractError("Final condition patient-window metadata model identity drifted.")
    row_keys = [str(row["condition_cache_key"]) for row in metadata]
    cache = ResumableEmbeddingCache(
        root,
        row_keys=row_keys,
        dimension=int(cache_metadata["dimension"]),
        configuration_hash=str(cache_metadata["configuration_hash"]),
    )
    vectors = cache.load_complete()
    if (
        len(metadata) != vectors.shape[0]
        or summary.get("rows") != len(metadata)
        or summary.get("dimension") != vectors.shape[1]
        or summary.get("metadata_sha256") != file_sha256(metadata_path)
        or summary.get("cache_complete_sha256") != file_sha256(cache.complete_path)
        or any(row.get("mirrors_embedding_row") is not True for row in metadata)
    ):
        raise ContractError("Final condition embedding/metadata one-to-one contract failed.")
    norms = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0.0):
        raise ContractError("Final condition cache contains non-unit embeddings.")
    return metadata, vectors
