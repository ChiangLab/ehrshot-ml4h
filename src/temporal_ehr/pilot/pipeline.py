"""DGX-only orchestration for the isolated complete three-task M1 pilot."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.history import iter_strict_histories
from temporal_ehr.experiments import PilotAuthorization, load_pilot_authorization, window_days_for_task
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, atomic_json, file_sha256
from temporal_ehr.pilot.evaluation import (
    calibration_summary,
    fit_logistic_readout,
    metric_set,
    paired_bootstrap_interval,
)
from temporal_ehr.pilot.index import finalize_indices, prepare_instance
from temporal_ehr.pilot.model import LocalQwenEncoder, ModelIdentity, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash
from temporal_ehr.serialization.packing import load_qwen_tokenizer
from temporal_ehr.serialization.render import load_code_descriptions


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ContractError(f"Refusing to write empty private Parquet: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_spec(config_path: Path, repository_root: Path) -> PilotAuthorization:
    return load_pilot_authorization(config_path, repository_root=repository_root)


def _model_identity(spec: PilotAuthorization, model_root: Path) -> ModelIdentity:
    directory = model_root / spec.model_directory_name
    return load_model_identity(directory, expected_repository=spec.model_repository)


def _pilot_git_commit() -> str:
    value = os.environ.get("PILOT_GIT_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ContractError("PILOT_GIT_COMMIT must be the full source commit recorded by the Slurm wrapper.")
    return value


def _validate_stage_identity(
    payload: Mapping[str, Any],
    *,
    stage: str,
    spec: PilotAuthorization,
    identity: ModelIdentity,
) -> None:
    expected = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "model_repository": identity.repository,
        "model_revision_hash": stable_hash(identity.revision),
        "model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "source_git_commit": _pilot_git_commit(),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ContractError(f"Pilot {stage} summary identity or completion state drifted.")


def _validate_pilot_cohort(spec: PilotAuthorization) -> dict[str, Any]:
    if not spec.cohort_path.is_file():
        raise ContractError("Authorized pilot cohort is missing.")
    connection = duckdb.connect()
    try:
        tasks = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT task FROM read_parquet(?)", [str(spec.cohort_path)]
            ).fetchall()
        }
        duplicate_instances = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                  SELECT instance_key FROM read_parquet(?) GROUP BY instance_key HAVING count(*) != 1
                )
                """,
                [str(spec.cohort_path)],
            ).fetchone()[0]
        )
        repeated_patient_task_split = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                  SELECT subject_id, task, split FROM read_parquet(?)
                  GROUP BY subject_id, task, split HAVING count(*) != 1
                )
                """,
                [str(spec.cohort_path)],
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT task, split, count(*) AS instances, avg(CAST(target AS DOUBLE)) AS prevalence
            FROM read_parquet(?)
            WHERE split IN ('train', 'validation')
            GROUP BY task, split ORDER BY task, split
            """,
            [str(spec.cohort_path)],
        ).fetchall()
    finally:
        connection.close()
    if tasks != set(spec.tasks):
        raise ContractError("Pilot cohort task set does not exactly match the authorized PILOT panel.")
    if duplicate_instances or repeated_patient_task_split:
        raise ContractError("Pilot cohort violates unique instance or one-patient-task-split contracts.")
    observed = {(str(task), str(split)) for task, split, _, _ in rows}
    expected = {(task, split) for task in spec.tasks for split in spec.splits}
    if observed != expected:
        raise ContractError("Pilot cohort lacks complete train/validation task coverage.")
    return {
        "cohort_sha256": file_sha256(spec.cohort_path),
        "groups": [
            {"task": str(task), "split": str(split), "instances": int(count), "prevalence": float(prevalence)}
            for task, split, count, prevalence in rows
        ],
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def _representation_label(row: Mapping[str, Any]) -> str:
    seed = row.get("perturbation_seed")
    return str(row["representation"]) if seed is None else f"{row['representation']}:seed={seed}"


def _preparation_coverage(
    indices: Sequence[Mapping[str, Any]], texts: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Aggregate private row-level packing coverage without exposing identifiers or timestamps."""
    index_groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in indices:
        index_groups[str(row["representation_key"])].append(row)
    text_by_key = {str(row["representation_key"]): row for row in texts}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for representation_key, rows in index_groups.items():
        first = rows[0]
        label = _representation_label(first)
        text = text_by_key.get(representation_key)
        eligible = any(row["retention_status"] != "INELIGIBLE" for row in rows)
        if eligible != (text is not None):
            raise ContractError("Patient-window eligibility does not align with the prepared text table.")
        retained = [row for row in rows if row["retention_status"] in {"RETAINED", "PARTIAL"}]
        grouped[label].append(
            {
                "eligible": eligible,
                "source_windows": len(rows),
                "retained_windows": len(retained),
                "retained_events": sum(int(row["retained_event_count"]) for row in retained),
                "source_events": sum(int(row["event_count"]) for row in rows),
                "token_count": int(text["token_count"]) if text is not None else None,
                "truncated": bool(text["truncated"]) if text is not None else None,
                "retained_historical_span_days": (
                    float(text["retained_historical_span_days"]) if text is not None else None
                ),
            }
        )
    coverage: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(grouped.items()):
        eligible_rows = [row for row in rows if row["eligible"]]
        coverage[label] = {
            "source_histories": len(rows),
            "eligible_histories": len(eligible_rows),
            "ineligible_histories": len(rows) - len(eligible_rows),
            "source_windows": sum(row["source_windows"] for row in eligible_rows),
            "retained_windows": sum(row["retained_windows"] for row in eligible_rows),
            "source_events": sum(row["source_events"] for row in eligible_rows),
            "retained_events": sum(row["retained_events"] for row in eligible_rows),
            "truncated_histories": sum(bool(row["truncated"]) for row in eligible_rows),
        }
        if eligible_rows:
            coverage[label]["token_count"] = _numeric_summary(
                [float(row["token_count"]) for row in eligible_rows]
            )
            coverage[label]["retained_historical_span_days"] = _numeric_summary(
                [float(row["retained_historical_span_days"]) for row in eligible_rows]
            )
    return coverage


def _assert_content_matching(texts: Sequence[Mapping[str, Any]]) -> None:
    full_names = {
        "C0_order_only",
        "C1_correct_compact_time",
        "C2_reversed_earlier_gaps",
        "C3_shuffled_earlier_time_labels",
    }
    by_instance: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in texts:
        if row["representation"] in full_names:
            by_instance[str(row["instance_key"])].append(row)
    if not by_instance:
        raise ContractError("No complete-history pilot texts were prepared.")
    for rows in by_instance.values():
        if len({row["source_content_hash"] for row in rows}) != 1:
            raise ContractError("C0-C3 source clinical content differs within one pilot instance.")
        if len({row["retained_content_hash"] for row in rows}) != 1:
            raise ContractError("C0-C3 retained content differs within one pilot instance.")


def prepare_pilot(
    *,
    repository_root: Path,
    config_path: Path,
    event_path: Path,
    asset_root: Path,
    model_root: Path,
) -> dict[str, Any]:
    """Prepare all private texts and patient-window indices; no model forward pass."""
    spec = _load_spec(config_path, repository_root)
    cohort = _validate_pilot_cohort(spec)
    identity = _model_identity(spec, model_root)
    source_git_commit = _pilot_git_commit()
    tokenizer = load_qwen_tokenizer(identity.directory)
    descriptions = load_code_descriptions(asset_root)
    event_data_sha256 = file_sha256(event_path)
    descriptions_hash = stable_hash(descriptions)
    registry = yaml.safe_load((repository_root / "configs/tasks.yaml").read_text(encoding="utf-8"))
    config_hash = stable_hash(spec.raw)
    task_summaries: list[dict[str, Any]] = []
    for task in spec.tasks:
        task_root = spec.private_root / "prepared" / task
        complete_path = task_root / "COMPLETE.json"
        task_input_hash = stable_hash(
            {
                "config_hash": config_hash,
                "cohort_sha256": cohort["cohort_sha256"],
                "event_data_sha256": event_data_sha256,
                "descriptions_hash": descriptions_hash,
                "model_hash": identity.artifact_hash,
                "tokenizer_hash": identity.tokenizer_hash,
                "task": task,
                "source_git_commit": source_git_commit,
            }
        )
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if complete.get("task_input_hash") != task_input_hash:
                raise ContractError(f"Completed pilot preparation drifted for task {task}.")
            index_path = task_root / "patient_window_index.parquet"
            texts_path = task_root / "texts.parquet"
            if not index_path.is_file() or not texts_path.is_file():
                raise ContractError(f"Completed pilot preparation is missing artifacts for task {task}.")
            if complete.get("patient_window_index_sha256") != file_sha256(index_path):
                raise ContractError(f"Completed patient-window index drifted for task {task}.")
            if complete.get("texts_sha256") != file_sha256(texts_path):
                raise ContractError(f"Completed prepared texts drifted for task {task}.")
            task_summaries.append(complete["aggregate_summary"])
            continue
        index_rows: list[dict[str, Any]] = []
        text_rows: list[dict[str, Any]] = []
        source_counts: list[int] = []
        deduplicated_counts: list[int] = []
        for instance in iter_strict_histories(
            event_path,
            spec.cohort_path,
            tasks=(task,),
            splits=spec.splits,
        ):
            prepared = prepare_instance(
                instance,
                experiment_id=spec.experiment_id,
                window_days=window_days_for_task(task, registry),
                c3_seeds=spec.c3_seeds,
                descriptions=descriptions,
                tokenizer=tokenizer,
                instruction=spec.instructions[task],
                budget=spec.token_budget,
                model_hash=identity.artifact_hash,
                tokenizer_hash=identity.tokenizer_hash,
            )
            index_rows.extend(prepared.index_rows)
            text_rows.extend(prepared.text_rows)
            source_counts.append(prepared.source_event_count)
            deduplicated_counts.append(prepared.deduplicated_event_count)
        indices, texts = finalize_indices(
            index_rows,
            text_rows,
            experiment_id=spec.experiment_id,
            model_hash=identity.artifact_hash,
            tokenizer_hash=identity.tokenizer_hash,
            instructions=spec.instructions,
            budget=spec.token_budget,
        )
        if {row["task"] for row in texts} != {task} or {row["split"] for row in texts} != set(spec.splits):
            raise ContractError("Prepared text table task/split alignment failed.")
        _assert_content_matching(texts)
        c1_rows = [row for row in texts if row["representation"] == "C1_correct_compact_time"]
        expected_instances = sum(
            int(group["instances"]) for group in cohort["groups"] if group["task"] == task
        )
        if len(source_counts) != expected_instances or len(c1_rows) != expected_instances:
            raise ContractError("Prepared pilot instances do not align one-to-one with the cohort.")
        index_path = task_root / "patient_window_index.parquet"
        texts_path = task_root / "texts.parquet"
        _atomic_parquet(index_path, indices)
        _atomic_parquet(texts_path, texts)
        representations = defaultdict(int)
        for row in texts:
            key = row["representation"]
            if row["perturbation_seed"] is not None:
                key = f"{key}:seed={row['perturbation_seed']}"
            representations[key] += 1
        summary = {
            "task": task,
            "source_instances": len(source_counts),
            "index_rows": len(indices),
            "eligible_text_rows": len(texts),
            "representations": dict(sorted(representations.items())),
            "source_event_count": _numeric_summary(source_counts),
            "deduplicated_event_count": _numeric_summary(deduplicated_counts),
            "token_count": _numeric_summary([row["token_count"] for row in texts]),
            "truncated_count": sum(bool(row["truncated"]) for row in texts),
            "source_historical_span_days": _numeric_summary(
                [float(row["source_historical_span_days"]) for row in c1_rows]
            ),
            "representation_coverage": _preparation_coverage(indices, texts),
            "c2_perturbable_histories": sum(
                row["representation"] == "C2_reversed_earlier_gaps" for row in texts
            ),
            "c3_perturbable_histories_by_seed": {
                str(seed): sum(
                    row["representation"] == "C3_shuffled_earlier_time_labels"
                    and row["perturbation_seed"] == seed
                    for row in texts
                )
                for seed in spec.c3_seeds
            },
            "content_matching_passed": True,
            "test_split_used": False,
        }
        complete = {
            "state": "COMPLETE",
            "task": task,
            "task_input_hash": task_input_hash,
            "patient_window_index_sha256": file_sha256(index_path),
            "texts_sha256": file_sha256(texts_path),
            "aggregate_summary": summary,
        }
        atomic_json(complete_path, complete)
        task_summaries.append(summary)
    aggregate = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": config_hash,
        "source_git_commit": source_git_commit,
        "mode": "PILOT",
        "tasks": list(spec.tasks),
        "splits": list(spec.splits),
        "token_budget": spec.token_budget,
        "model_repository": identity.repository,
        "model_revision_hash": stable_hash(identity.revision),
        "model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "cohort_sha256": cohort["cohort_sha256"],
        "event_data_sha256": event_data_sha256,
        "descriptions_hash": descriptions_hash,
        "cohort_groups": cohort["groups"],
        "task_summaries": task_summaries,
        "test_predictions_or_performance_used": False,
    }
    text = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    destination = spec.aggregate_root / "prepare_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return aggregate


def _read_task_texts(spec: PilotAuthorization, task: str) -> list[dict[str, Any]]:
    task_root = spec.private_root / "prepared" / task
    complete_path = task_root / "COMPLETE.json"
    texts_path = task_root / "texts.parquet"
    if not complete_path.is_file() or not texts_path.is_file():
        raise ContractError(f"Task {task} preparation is incomplete.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("texts_sha256") != file_sha256(texts_path):
        raise ContractError(f"Task {task} prepared text hash drifted.")
    rows = pq.read_table(texts_path).to_pylist()
    rows.sort(key=lambda row: int(row["embedding_row_index"]))
    if [int(row["embedding_row_index"]) for row in rows] != list(range(len(rows))):
        raise ContractError("Prepared embedding row indices are not contiguous and deterministic.")
    if len({row["cache_key"] for row in rows}) != len(rows):
        raise ContractError("Prepared embedding cache keys are not unique.")
    return rows


def benchmark_throughput(
    *, repository_root: Path,
    config_path: Path,
    model_root: Path,
) -> dict[str, Any]:
    """Benchmark short/median/near-limit private inputs without reading outcomes."""
    spec = _load_spec(config_path, repository_root)
    identity = _model_identity(spec, model_root)
    candidates: list[dict[str, Any]] = []
    for task in spec.tasks:
        rows = _read_task_texts(spec, task)
        ordered = sorted(rows, key=lambda row: int(row["token_count"]))
        for quantile, index in (("short", 0), ("median", len(ordered) // 2), ("near_limit", -1)):
            candidates.append(
                {
                    "task": task,
                    "quantile": quantile,
                    "text": ordered[index]["text"],
                    "tokens": ordered[index]["token_count"],
                }
            )
    encoder = LocalQwenEncoder(identity, max_seq_length=spec.token_budget)
    encoder.encode(["Synthetic warm-up record."], batch_size=1)
    encoder.reset_peak_memory()
    started = time.perf_counter()
    encoder.encode([row["text"] for row in candidates], batch_size=int(spec.raw["batch_size"]))
    elapsed = time.perf_counter() - started
    total_tokens = sum(int(row["tokens"]) for row in candidates)
    result = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "source_git_commit": _pilot_git_commit(),
        "model_repository": identity.repository,
        "model_revision_hash": stable_hash(identity.revision),
        "model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "token_budget": spec.token_budget,
        "warmup_policy": "one fully synthetic input excluded from timing",
        "sample_policy": "short, median, and near-limit prepared input per pilot task",
        "sample_count": len(candidates),
        "batch_size": int(spec.raw["batch_size"]),
        "elapsed_seconds": elapsed,
        "documents_per_second": len(candidates) / elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "peak_gpu_memory_bytes": encoder.peak_memory_bytes(),
        "device_name": encoder.device_name(),
        "failures": 0,
        "retries": 0,
        "patient_text_printed": False,
        "test_outcomes_used": False,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    destination = spec.aggregate_root / "throughput.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return result


def embed_task(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    task: str,
) -> dict[str, Any]:
    """Resume one deterministic task embedding cache and finalize only when every row is valid."""
    spec = _load_spec(config_path, repository_root)
    if task not in spec.tasks:
        raise ContractError(f"Task {task} is not in the authorized pilot.")
    identity = _model_identity(spec, model_root)
    source_git_commit = _pilot_git_commit()
    rows = _read_task_texts(spec, task)
    row_keys = [str(row["cache_key"]) for row in rows]
    texts_hash = file_sha256(spec.private_root / "prepared" / task / "texts.parquet")
    configuration_hash = stable_hash(
        {
            "config": spec.raw,
            "task": task,
            "source_git_commit": source_git_commit,
            "texts_sha256": texts_hash,
            "model_hash": identity.artifact_hash,
            "tokenizer_hash": identity.tokenizer_hash,
        }
    )
    encoder = LocalQwenEncoder(identity, max_seq_length=spec.token_budget)
    cache = ResumableEmbeddingCache(
        spec.private_root / "embeddings" / task,
        row_keys=row_keys,
        dimension=encoder.dimension,
        configuration_hash=configuration_hash,
    )
    attempt = cache.begin_attempt()
    pending = cache.pending_indices()
    batch_size = int(spec.raw["batch_size"])
    encoder.reset_peak_memory()
    started = time.perf_counter()
    for offset in range(0, len(pending), batch_size):
        indices = pending[offset : offset + batch_size]
        vectors = encoder.encode([str(rows[int(index)]["text"]) for index in indices], batch_size=batch_size)
        cache.write_rows(indices.tolist(), vectors)
    cache_complete_path = cache.finalize()
    attempt = cache.finish_attempt()
    elapsed = time.perf_counter() - started
    result = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "source_git_commit": source_git_commit,
        "task": task,
        "model_repository": identity.repository,
        "model_revision_hash": stable_hash(identity.revision),
        "model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "rows": len(rows),
        "rows_computed_this_attempt": len(pending),
        "attempt_number": attempt["attempt_count"],
        "elapsed_seconds": elapsed,
        "documents_per_second": len(pending) / elapsed if elapsed and len(pending) else None,
        "peak_gpu_memory_bytes": encoder.peak_memory_bytes(),
        "failures": attempt["failed_attempts"],
        "retries": attempt["retries"],
        "cache_complete": True,
        "cache_complete_sha256": file_sha256(cache_complete_path),
    }
    destination = spec.aggregate_root / "embedding" / f"{task}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _representation_rows(
    rows: Sequence[dict[str, Any]], representation: str, seed: int | None, split: str
) -> dict[str, dict[str, Any]]:
    return {
        str(row["instance_key"]): row
        for row in rows
        if row["representation"] == representation
        and row["perturbation_seed"] == seed
        and row["split"] == split
    }


def _fit_scope(
    *,
    task: str,
    scope: str,
    representations: Sequence[tuple[str, int | None]],
    rows: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    c_grid: Sequence[float],
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_split = {
        split: {
            (representation, seed): _representation_rows(rows, representation, seed, split)
            for representation, seed in representations
        }
        for split in ("train", "validation")
    }
    common = {
        split: set.intersection(*(set(mapping) for mapping in by_split[split].values()))
        for split in ("train", "validation")
    }
    if not common["train"] or not common["validation"]:
        raise ContractError(f"Pilot fitting scope {scope} has no aligned train/validation instances.")
    scope_report: dict[str, Any] = {
        "train_instances": len(common["train"]),
        "validation_instances": len(common["validation"]),
        "train_prevalence": float(
            np.mean(
                [
                    bool(by_split["train"][representations[0]][key]["target"])
                    for key in sorted(common["train"])
                ]
            )
        ),
        "validation_prevalence": float(
            np.mean(
                [
                    bool(by_split["validation"][representations[0]][key]["target"])
                    for key in sorted(common["validation"])
                ]
            )
        ),
        "representations": {},
    }
    private_predictions: list[dict[str, Any]] = []
    for representation, seed in representations:
        matrices: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        keys_by_split: dict[str, list[str]] = {}
        for split in ("train", "validation"):
            keys = sorted(common[split])
            keys_by_split[split] = keys
            selected = [by_split[split][(representation, seed)][key] for key in keys]
            matrices[split] = np.asarray(
                embeddings[[int(row["embedding_row_index"]) for row in selected]], dtype=np.float32
            )
            targets[split] = np.asarray([bool(row["target"]) for row in selected], dtype=np.int8)
        for split in ("train", "validation"):
            reference_targets = {
                tuple(bool(by_split[split][pair][key]["target"]) for key in keys_by_split[split])
                for pair in representations
            }
            if len(reference_targets) != 1:
                raise ContractError("C0-C3 scope patient/label alignment failed.")
        probability, fit_report = fit_logistic_readout(
            matrices["train"],
            targets["train"],
            matrices["validation"],
            targets["validation"],
            c_grid=c_grid,
            random_seed=random_seed,
        )
        label = representation if seed is None else f"{representation}:seed={seed}"
        scope_report["representations"][label] = fit_report
        for key, target, value in zip(keys_by_split["validation"], targets["validation"], probability, strict=True):
            private_predictions.append(
                {
                    "task": task,
                    "scope": scope,
                    "representation": representation,
                    "perturbation_seed": seed,
                    "instance_key": key,
                    "target": bool(target),
                    "probability": float(value),
                }
            )
    return scope_report, private_predictions


def _contrast(
    predictions: Sequence[dict[str, Any]],
    *,
    task: str,
    scope: str,
    first: tuple[str, int | None],
    second: tuple[str, int | None],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected = [row for row in predictions if row["task"] == task and row["scope"] == scope]
    mappings = {}
    for representation, perturbation_seed in (first, second):
        mappings[(representation, perturbation_seed)] = {
            row["instance_key"]: row
            for row in selected
            if row["representation"] == representation and row["perturbation_seed"] == perturbation_seed
        }
    if set(mappings[first]) != set(mappings[second]):
        raise ContractError("Paired contrast instance sets differ.")
    keys = sorted(mappings[first])
    targets = np.asarray([mappings[first][key]["target"] for key in keys], dtype=np.int8)
    if targets.tolist() != [mappings[second][key]["target"] for key in keys]:
        raise ContractError("Paired contrast labels differ.")
    return {
        "first": first[0] if first[1] is None else f"{first[0]}:seed={first[1]}",
        "second": second[0] if second[1] is None else f"{second[0]}:seed={second[1]}",
        "instances": len(keys),
        **paired_bootstrap_interval(
            targets,
            np.asarray([mappings[first][key]["probability"] for key in keys]),
            np.asarray([mappings[second][key]["probability"] for key in keys]),
            replicates=replicates,
            seed=seed,
        ),
    }


def fit_and_report(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
) -> dict[str, Any]:
    """Fit all pilot readouts and write private predictions plus one aggregate report."""
    spec = _load_spec(config_path, repository_root)
    identity = _model_identity(spec, model_root)
    prepare_summary = json.loads((spec.aggregate_root / "prepare_summary.json").read_text(encoding="utf-8"))
    throughput = json.loads((spec.aggregate_root / "throughput.json").read_text(encoding="utf-8"))
    _validate_stage_identity(prepare_summary, stage="preparation", spec=spec, identity=identity)
    _validate_stage_identity(throughput, stage="throughput", spec=spec, identity=identity)
    if prepare_summary.get("cohort_sha256") != file_sha256(spec.cohort_path):
        raise ContractError("Pilot cohort changed after preparation.")
    embedding_summaries = [
        json.loads((spec.aggregate_root / "embedding" / f"{task}.json").read_text(encoding="utf-8"))
        for task in spec.tasks
    ]
    for task, summary in zip(spec.tasks, embedding_summaries, strict=True):
        _validate_stage_identity(summary, stage=f"embedding task {task}", spec=spec, identity=identity)
        if summary.get("task") != task or summary.get("cache_complete") is not True:
            raise ContractError(f"Pilot embedding completion summary drifted for task {task}.")
    c_grid = tuple(float(value) for value in spec.raw["regularization_grid"])
    logistic_seed = int(spec.raw["logistic_seed"])
    bootstrap_seed = int(spec.raw["bootstrap_seed"])
    replicates = int(spec.raw["bootstrap_replicates"])
    all_private_predictions: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}
    for task_index, task in enumerate(spec.tasks):
        rows = _read_task_texts(spec, task)
        cache_metadata_path = spec.private_root / "embeddings" / task / "CACHE.json"
        cache_metadata = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
        cache = ResumableEmbeddingCache(
            spec.private_root / "embeddings" / task,
            row_keys=[str(row["cache_key"]) for row in rows],
            dimension=int(cache_metadata["dimension"]),
            configuration_hash=str(cache_metadata["configuration_hash"]),
        )
        embeddings = cache.load_complete()
        complete_representations = (
            ("B1_latest_window_only", None),
            ("B2_latest_code_occurrence", None),
            ("C0_order_only", None),
            ("C1_correct_compact_time", None),
        )
        scopes: dict[str, Sequence[tuple[str, int | None]]] = {
            "complete_history": complete_representations,
            "c2_perturbable": (
                ("C0_order_only", None),
                ("C1_correct_compact_time", None),
                ("C2_reversed_earlier_gaps", None),
            ),
            **{
                f"c3_seed_{seed}": (
                    ("C0_order_only", None),
                    ("C1_correct_compact_time", None),
                    ("C3_shuffled_earlier_time_labels", seed),
                )
                for seed in spec.c3_seeds
            },
        }
        scope_reports = {}
        task_predictions: list[dict[str, Any]] = []
        for scope, representations in scopes.items():
            scope_report, predictions = _fit_scope(
                task=task,
                scope=scope,
                representations=representations,
                rows=rows,
                embeddings=embeddings,
                c_grid=c_grid,
                random_seed=logistic_seed,
            )
            scope_reports[scope] = scope_report
            task_predictions.extend(predictions)
        complete_train = _representation_rows(rows, "C1_correct_compact_time", None, "train")
        complete_validation = _representation_rows(rows, "C1_correct_compact_time", None, "validation")
        train_prevalence = float(np.mean([bool(row["target"]) for row in complete_train.values()]))
        validation_targets = np.asarray(
            [bool(complete_validation[key]["target"]) for key in sorted(complete_validation)], dtype=np.int8
        )
        prevalence_probability = np.full(len(validation_targets), train_prevalence)
        b4 = {
            "train_instances": len(complete_train),
            "validation_instances": len(complete_validation),
            "training_prevalence": train_prevalence,
            "metrics": metric_set(validation_targets, prevalence_probability),
            "calibration": calibration_summary(validation_targets, prevalence_probability),
        }
        count_report_path = repository_root / "reports/generated/count_baseline/validation_results.json"
        count_predictions_path = (
            repository_root / f"outputs/private/count_baseline/{task}/validation_predictions.parquet"
        )
        if not count_report_path.is_file() or not count_predictions_path.is_file():
            raise ContractError("Governed B3 count-baseline report/predictions are required for the complete pilot.")
        count_report = json.loads(count_report_path.read_text(encoding="utf-8"))
        if count_report.get("status") != "PASS" or count_report.get("test_split_inspected") is not False:
            raise ContractError("Governed B3 count-baseline report is not an eligible train/validation result.")
        if task not in count_report.get("tasks", {}):
            raise ContractError(f"Governed B3 count-baseline report is missing task {task}.")
        count_rows = pq.read_table(count_predictions_path).to_pylist()
        count_map = {str(row["instance_key"]): row for row in count_rows}
        if len(count_map) != len(count_rows):
            raise ContractError("B3 count predictions contain duplicate pilot instance keys.")
        if set(count_map) != set(complete_validation):
            raise ContractError("B3 count predictions do not align one-to-one with pilot validation instances.")
        for key, row in count_map.items():
            if bool(row["target"]) != bool(complete_validation[key]["target"]):
                raise ContractError("B3 count labels drifted from pilot labels.")
        validation_keys = sorted(complete_validation)
        for key, target, probability in zip(
            validation_keys,
            validation_targets,
            prevalence_probability,
            strict=True,
        ):
            task_predictions.append(
                {
                    "task": task,
                    "scope": "complete_history",
                    "representation": "B4_training_prevalence",
                    "perturbation_seed": None,
                    "instance_key": key,
                    "target": bool(target),
                    "probability": float(probability),
                }
            )
            task_predictions.append(
                {
                    "task": task,
                    "scope": "complete_history",
                    "representation": "B3_governed_code_count",
                    "perturbation_seed": None,
                    "instance_key": key,
                    "target": bool(target),
                    "probability": float(count_map[key]["probability"]),
                }
            )
        b3_report = dict(count_report["tasks"][task])
        b3_report["calibration"] = calibration_summary(
            validation_targets,
            np.asarray([float(count_map[key]["probability"]) for key in validation_keys]),
        )
        contrasts = [
            _contrast(
                task_predictions,
                task=task,
                scope="complete_history",
                first=("C1_correct_compact_time", None),
                second=(representation, None),
                replicates=replicates,
                seed=bootstrap_seed + task_index,
            )
            for representation in (
                "B4_training_prevalence",
                "B3_governed_code_count",
                "B1_latest_window_only",
                "B2_latest_code_occurrence",
                "C0_order_only",
            )
        ]
        contrasts.append(
            _contrast(
                task_predictions,
                task=task,
                scope="c2_perturbable",
                first=("C1_correct_compact_time", None),
                second=("C2_reversed_earlier_gaps", None),
                replicates=replicates,
                seed=bootstrap_seed + 100 + task_index,
            )
        )
        c3_contrasts = [
            _contrast(
                task_predictions,
                task=task,
                scope=f"c3_seed_{seed}",
                first=("C1_correct_compact_time", None),
                second=("C3_shuffled_earlier_time_labels", seed),
                replicates=replicates,
                seed=bootstrap_seed + int(seed) + task_index,
            )
            for seed in spec.c3_seeds
        ]
        c3_estimates = [item["estimate"] for item in c3_contrasts]
        task_reports[task] = {
            "scopes": scope_reports,
            "B4_training_prevalence": b4,
            "B3_governed_code_count": b3_report,
            "paired_contrasts": contrasts,
            "c3_seed_contrasts": c3_contrasts,
            "c3_seed_summary": {
                "mean_auroc_difference": float(np.mean(c3_estimates)),
                "standard_deviation": float(np.std(c3_estimates, ddof=1)),
                "seeds": list(spec.c3_seeds),
            },
        }
        all_private_predictions.extend(task_predictions)
    prediction_keys = [
        (
            row["task"],
            row["scope"],
            row["representation"],
            row["perturbation_seed"],
            row["instance_key"],
        )
        for row in all_private_predictions
    ]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise ContractError("Pilot private predictions are not one-to-one within task/scope/condition.")
    predictions_path = spec.private_root / "fit" / "validation_predictions.parquet"
    _atomic_parquet(predictions_path, all_private_predictions)
    report = {
        "status": "PASS",
        "protocol_decision": "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "source_git_commit": _pilot_git_commit(),
        "mode": "PILOT",
        "scientific_scope": {
            "tasks": list(spec.tasks),
            "splits": list(spec.splits),
            "model": spec.model_repository,
            "model_revision_hash": stable_hash(identity.revision),
            "model_artifact_hash": identity.artifact_hash,
            "tokenizer_hash": identity.tokenizer_hash,
            "token_budget": spec.token_budget,
            "approach": "M1 direct serialization only",
            "baselines": ["B4", "B3", "B1", "B2"],
            "conditions": ["C0", "C1", "C2", "C3"],
            "c3_seeds": list(spec.c3_seeds),
        },
        "prepare": prepare_summary,
        "throughput": throughput,
        "embedding_tasks": embedding_summaries,
        "tasks": task_reports,
        "private_predictions_sha256": file_sha256(predictions_path),
        "private_prediction_rows": len(all_private_predictions),
        "content_matching_passed": True,
        "final_panel_used_for_pilot": False,
        "test_predictions_or_performance_used": False,
        "m2_run": False,
        "m3_run": False,
        "qwen_8b_run": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    destination = spec.aggregate_root / "pilot_results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return report
