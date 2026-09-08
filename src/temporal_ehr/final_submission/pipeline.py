"""Deterministic held-out cohort, serialization, and frozen M1 embeddings."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.history import iter_strict_histories
from temporal_ehr.experiments import M1SubmissionAuthorization, window_days_for_task
from temporal_ehr.final_m1.cohort import (
    _split_map,
    _validated_unique_times,
    _zip_member_sha256,
    selection_rank,
)
from temporal_ehr.final_m1.embedding import _condition_rows, _embed_condition
from temporal_ehr.final_m1.foundation import (
    _assert_content_and_label_only_perturbations,
    _metadata_rows,
    _task_prepare_summary,
    _write_immutable_json,
)
from temporal_ehr.final_submission.protocol import (
    begin_test_access,
    complete_test_access,
    load_frozen_manifest,
)
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.pilot.index import finalize_indices, prepare_instance
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity
from temporal_ehr.serialization.blocks import deduplicate_events, event_from_mapping
from temporal_ehr.serialization.integrity import stable_hash
from temporal_ehr.serialization.packing import load_qwen_tokenizer
from temporal_ehr.serialization.render import load_code_descriptions


def _cohort_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("instance_key", pa.string(), nullable=False),
            pa.field("selection_hash", pa.string(), nullable=False),
            pa.field("subject_id", pa.int64(), nullable=False),
            pa.field("prediction_time", pa.timestamp("us"), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("family", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("target", pa.bool_(), nullable=False),
            pa.field("dataset_release_id", pa.string(), nullable=False),
            pa.field("selection_seed", pa.int64(), nullable=False),
            pa.field("eligible_prediction_time_count", pa.int64(), nullable=False),
            pa.field("selected_prediction_time_index", pa.int64(), nullable=False),
        ]
    )


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema | None = None) -> None:
    if path.exists() or not rows:
        raise ContractError(f"Refusing to overwrite or write empty held-out Parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table = pa.Table.from_pylist(rows, schema=schema) if schema else pa.Table.from_pylist(rows)
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _private_cohort_summary(
    rows: list[dict[str, Any]],
    spec: M1SubmissionAuthorization,
) -> dict[str, Any]:
    counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counter = counts[str(row["task"])]
        counter["instances"] += 1
        counter["positive_instances"] += int(bool(row["target"]))
    groups = []
    for task in spec.tasks:
        count = counts[task]
        total = int(count["instances"])
        positives = int(count["positive_instances"])
        if total < 1 or positives < 1 or positives >= total:
            raise ContractError(f"Held-out cohort for {task} must contain both outcome classes.")
        groups.append(
            {
                "task": task,
                "split": "test",
                "instances": total,
                "positive_instances": positives,
                "negative_instances": total - positives,
                "prevalence": positives / total,
            }
        )
    return {
        "groups": groups,
        "selected_instances": len(rows),
        "maximum_instances_per_patient_task": 1,
        "split": "test",
    }


def build_test_cohort(
    *,
    repository_root: Path,
    config_path: Path,
    asset_root: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Open held-out labels only after freeze and build the outcome-independent test cohort."""
    root = repository_root.resolve()
    spec, manifest = load_frozen_manifest(root, config_path, source_git_commit)
    begin_test_access(spec, manifest)
    output = spec.test_cohort_path
    complete_path = output.parent / "COMPLETE.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            complete.get("state") != "COMPLETE"
            or complete.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
            or not output.is_file()
            or complete.get("cohort_sha256") != file_sha256(output)
        ):
            raise ContractError("Existing held-out cohort is stale or hash-mismatched.")
        complete_test_access(
            spec,
            manifest,
            cohort_sha256=str(complete["cohort_sha256"]),
        )
        return dict(complete["private_summary"])
    if output.exists():
        raise ContractError("A held-out cohort exists without a valid completion record.")

    registry = yaml.safe_load((root / "configs/tasks.yaml").read_text(encoding="utf-8"))
    split_map = _split_map(asset_root, registry["split_mapping"])
    release = spec.validation.raw["dataset_release"]
    release_id = str(release["release_id"])
    seed = int(spec.validation.raw["selection"]["frozen_seed"])
    minimum = int(spec.validation.raw["minimum_report_group_size"])
    selected: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    duplicate_audit: list[dict[str, Any]] = []
    for task_name in spec.tasks:
        task = registry["tasks"][task_name]
        field = str(task["label_field"])
        suffix = f"labels/{task_name}/labels.parquet"
        from temporal_ehr.data.zip_assets import read_parquet_member

        table = read_parquet_member(asset_root, suffix, ["subject_id", "prediction_time", field])
        source_hashes[task_name] = _zip_member_sha256(asset_root, suffix)
        grouped, audit = _validated_unique_times(
            table,
            task_name=task_name,
            task=task,
            split_map=split_map,
            allowed_splits=("test",),
            minimum_report_group_size=minimum,
        )
        duplicate_audit.extend(audit)
        for (subject_id, split), eligible in grouped.items():
            ranked = [
                (
                    selection_rank(
                        dataset_release_id=release_id,
                        task=task_name,
                        split=split,
                        subject_id=subject_id,
                        frozen_seed=seed,
                        sorted_prediction_time_index=index,
                    ),
                    index,
                    prediction_time,
                    target,
                )
                for index, (prediction_time, target) in enumerate(eligible)
            ]
            rank, index, prediction_time, target = min(ranked, key=lambda item: item[0])
            selected.append(
                {
                    "instance_key": stable_hash(
                        {
                            "experiment_id": spec.experiment_id,
                            "selection_hash": rank,
                            "official_prediction_time": prediction_time,
                        }
                    ),
                    "selection_hash": rank,
                    "subject_id": subject_id,
                    "prediction_time": prediction_time,
                    "task": task_name,
                    "family": str(task["family"]),
                    "split": "test",
                    "target": bool(target),
                    "dataset_release_id": release_id,
                    "selection_seed": seed,
                    "eligible_prediction_time_count": len(eligible),
                    "selected_prediction_time_index": index,
                }
            )
    selected.sort(key=lambda row: (row["task"], row["instance_key"]))
    if len({(row["subject_id"], row["task"]) for row in selected}) != len(selected):
        raise ContractError("Held-out cohort contains more than one instance per patient/task.")
    if {row["task"] for row in selected} != set(spec.tasks) or {row["split"] for row in selected} != {"test"}:
        raise ContractError("Held-out cohort task/split coverage drifted.")
    _atomic_parquet(output, selected, _cohort_schema())
    summary = _private_cohort_summary(selected, spec)
    complete = {
        "state": "COMPLETE",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
        "cohort_sha256": file_sha256(output),
        "cohort_rows": len(selected),
        "source_label_artifact_hashes": source_hashes,
        "duplicate_prediction_time_audit": duplicate_audit,
        "selection_uses_outcome": False,
        "private_summary": summary,
    }
    atomic_json(complete_path, complete)
    complete_test_access(spec, manifest, cohort_sha256=str(complete["cohort_sha256"]))
    return summary


def prepare_test_task(
    *,
    repository_root: Path,
    config_path: Path,
    event_path: Path,
    asset_root: Path,
    model_root: Path,
    source_git_commit: str,
    task: str,
) -> dict[str, Any]:
    """Serialize one held-out task under the frozen M1 representation contract."""
    root = repository_root.resolve()
    spec, manifest = load_frozen_manifest(root, config_path, source_git_commit)
    if task not in spec.tasks:
        raise ContractError("Held-out preparation task is outside the locked panel.")
    access = spec.private_root / "test_access" / "COMPLETE.json"
    cohort_complete_path = spec.test_cohort_path.parent / "COMPLETE.json"
    if not access.is_file() or not cohort_complete_path.is_file():
        raise ContractError("Held-out task preparation requires completed one-time cohort access.")
    cohort_complete = json.loads(cohort_complete_path.read_text(encoding="utf-8"))
    if cohort_complete.get("cohort_sha256") != file_sha256(spec.test_cohort_path):
        raise ContractError("Held-out cohort hash drifted before serialization.")
    if file_sha256(event_path) != spec.validation.raw["dataset_release"]["event_data_sha256"]:
        raise ContractError("Governed event data hash drifted from the locked release.")
    task_root = spec.private_root / "prepared" / task
    complete_path = task_root / "COMPLETE.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        required = {
            "patient_window_index_sha256": task_root / "patient_window_index.parquet",
            "patient_window_metadata_sha256": task_root / "patient_window_metadata.parquet",
            "texts_sha256": task_root / "texts.parquet",
        }
        if (
            complete.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
            or any(not path.is_file() or complete.get(field) != file_sha256(path) for field, path in required.items())
        ):
            raise ContractError("Existing held-out preparation is stale or hash-mismatched.")
        return dict(complete["private_summary"])
    if task_root.exists():
        raise ContractError("Partial held-out task preparation exists; preserve it for recovery review.")

    registry = yaml.safe_load((root / "configs/tasks.yaml").read_text(encoding="utf-8"))
    identity = load_model_identity(
        model_root / spec.validation.primary_model_directory_name,
        expected_repository=spec.validation.primary_model_repository,
    )
    tokenizer = load_qwen_tokenizer(identity.directory)
    descriptions = load_code_descriptions(asset_root)
    index_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    source_counts: list[int] = []
    deduplicated_counts: list[int] = []
    recurrence_counts: list[int] = []
    for instance in iter_strict_histories(
        event_path,
        spec.test_cohort_path,
        tasks=(task,),
        splits=("test",),
        allow_test=True,
    ):
        if any(row["time"] >= instance.prediction_time for row in instance.events):
            raise ContractError("Held-out serializer retained an event at or after prediction time.")
        events = tuple(event_from_mapping(row) for row in instance.events)
        deduplicated, _ = deduplicate_events(events)
        prepared = prepare_instance(
            instance,
            experiment_id=spec.experiment_id,
            window_days=window_days_for_task(task, registry),
            c3_seeds=spec.validation.c3_seeds,
            descriptions=descriptions,
            tokenizer=tokenizer,
            instruction=spec.validation.instructions[task],
            budget=spec.validation.token_budget,
            model_hash=identity.artifact_hash,
            tokenizer_hash=identity.tokenizer_hash,
        )
        index_rows.extend(prepared.index_rows)
        text_rows.extend(prepared.text_rows)
        source_counts.append(len(events))
        deduplicated_counts.append(len(deduplicated))
        recurrence_counts.append(len(deduplicated) - len({event.code for event in deduplicated}))
    indices, texts = finalize_indices(
        index_rows,
        text_rows,
        experiment_id=spec.experiment_id,
        model_hash=identity.artifact_hash,
        tokenizer_hash=identity.tokenizer_hash,
        instructions=spec.validation.instructions,
        budget=spec.validation.token_budget,
    )
    _assert_content_and_label_only_perturbations(indices, texts)
    expected = sum(
        int(group["instances"])
        for group in cohort_complete["private_summary"]["groups"]
        if group["task"] == task
    )
    c1_count = sum(row["representation"] == "C1_correct_compact_time" for row in texts)
    if len(source_counts) != expected or c1_count != expected:
        raise ContractError("Held-out C1 preparation is not one-to-one with the cohort.")
    metadata = _metadata_rows(texts)
    summary = _task_prepare_summary(
        task=task,
        indices=indices,
        texts=texts,
        source_counts=source_counts,
        deduplicated_counts=deduplicated_counts,
        recurrence_counts=recurrence_counts,
    )
    summary["test_split_used"] = True
    task_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{task}.", dir=task_root.parent))
    try:
        paths = {
            "patient_window_index_sha256": temporary / "patient_window_index.parquet",
            "patient_window_metadata_sha256": temporary / "patient_window_metadata.parquet",
            "texts_sha256": temporary / "texts.parquet",
        }
        pq.write_table(pa.Table.from_pylist(indices), paths["patient_window_index_sha256"], compression="zstd")
        pq.write_table(pa.Table.from_pylist(metadata), paths["patient_window_metadata_sha256"], compression="zstd")
        pq.write_table(pa.Table.from_pylist(texts), paths["texts_sha256"], compression="zstd")
        complete = {
            "state": "COMPLETE",
            "experiment_id": spec.experiment_id,
            "task": task,
            "source_git_commit": source_git_commit,
            "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
            **{field: file_sha256(path) for field, path in paths.items()},
            "private_summary": summary,
        }
        atomic_json(temporary / "COMPLETE.json", complete)
        temporary.replace(task_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return summary


def embed_test_task(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    source_git_commit: str,
    task: str,
) -> dict[str, Any]:
    """Embed every frozen M1 held-out representation with the primary 0.6B model only."""
    root = repository_root.resolve()
    spec, manifest = load_frozen_manifest(root, config_path, source_git_commit)
    if task not in spec.tasks:
        raise ContractError("Held-out embedding task is outside the locked panel.")
    prepared_root = spec.private_root / "prepared" / task
    complete_path = prepared_root / "COMPLETE.json"
    texts_path = prepared_root / "texts.parquet"
    if not complete_path.is_file() or not texts_path.is_file():
        raise ContractError("Held-out embedding requires completed task preparation.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
        or complete.get("texts_sha256") != file_sha256(texts_path)
    ):
        raise ContractError("Held-out prepared text provenance drifted.")
    rows = pq.read_table(texts_path).to_pylist()
    grouped = _condition_rows(rows)
    identity = load_model_identity(
        model_root / spec.validation.primary_model_directory_name,
        expected_repository=spec.validation.primary_model_repository,
    )
    encoder = LocalQwenEncoder(identity, max_seq_length=spec.validation.token_budget)
    summaries = [
        _embed_condition(
            spec=spec.validation,
            task=task,
            label=label,
            rows=condition_rows,
            role="primary",
            identity=identity,
            encoder=encoder,
            batch_size=int(spec.validation.raw["models"]["primary"]["batch_size"]),
            source_git_commit=source_git_commit,
            prepared_texts_sha256=file_sha256(texts_path),
            artifact_experiment_id=spec.experiment_id,
            cache_base=spec.private_root / "embeddings",
            test_split_used=True,
        )
        for label, condition_rows in grouped.items()
    ]
    report = {
        "status": "PASS",
        "protocol_decision": "M1_HELD_OUT_EMBEDDINGS_COMPLETE_RESULTS_NOT_YET_REPORTED",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
        "task": task,
        "model_repository": identity.repository,
        "model_artifact_hash": identity.artifact_hash,
        "condition_count": len(summaries),
        "conditions": [summary["condition"] for summary in summaries],
        "rows_by_condition": {summary["condition"]: summary["rows"] for summary in summaries},
        "full_dimension": True,
        "l2_normalized_float32": True,
        "immutable_verified_cache_reuse": True,
        "test_split_accessed": True,
        "test_performance_reported": False,
    }
    _write_immutable_json(spec.aggregate_root / "embedding" / f"{task}.json", report, aggregate=True)
    return report
