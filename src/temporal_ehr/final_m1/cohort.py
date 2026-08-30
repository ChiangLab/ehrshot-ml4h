"""Immutable final-panel cohort selection from official EHRSHOT prediction times."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.task_registry import binary_target, validate_domain
from temporal_ehr.data.zip_assets import locate_member, read_parquet_member
from temporal_ehr.experiments import FinalM1Authorization
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.serialization.integrity import stable_hash


def selection_rank(
    *,
    dataset_release_id: str,
    task: str,
    split: str,
    subject_id: int,
    frozen_seed: int,
    sorted_prediction_time_index: int,
) -> str:
    """Rank a sorted official time without using time value, outcome, history, or model result."""
    if sorted_prediction_time_index < 0:
        raise ValueError("Sorted prediction-time index cannot be negative.")
    return stable_hash(
        {
            "dataset_release_id": dataset_release_id,
            "task": task,
            "split": split,
            "patient": int(subject_id),
            "frozen_seed": int(frozen_seed),
            "sorted_eligible_prediction_time_index": int(sorted_prediction_time_index),
        }
    )


def _split_map(asset_root: Path, split_mapping: Mapping[str, str]) -> dict[int, str]:
    table = read_parquet_member(asset_root, "metadata/subject_splits.parquet", ["subject_id", "split"])
    result: dict[int, str] = {}
    for subject_id, raw_split in zip(table["subject_id"].to_pylist(), table["split"].to_pylist(), strict=True):
        if int(subject_id) in result:
            raise ContractError("Canonical split artifact contains a duplicate patient.")
        if raw_split not in split_mapping:
            raise ContractError(f"Unexpected official split value: {raw_split!r}")
        result[int(subject_id)] = str(split_mapping[raw_split])
    return result


def _zip_member_sha256(asset_root: Path, suffix: str) -> str:
    archive, member = locate_member(asset_root, suffix)
    digest = hashlib.sha256()
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    if path.exists():
        raise ContractError(f"Refusing to overwrite an existing final cohort artifact: {path.name}")
    if not rows:
        raise ContractError("Refusing to write an empty final cohort.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validated_unique_times(
    table: pa.Table,
    *,
    task_name: str,
    task: Mapping[str, Any],
    split_map: Mapping[int, str],
    allowed_splits: tuple[str, ...],
) -> dict[tuple[int, str], list[tuple[Any, bool]]]:
    """Collapse outcome-identical duplicate times and reject ambiguous official labels."""
    field = str(task["label_field"])
    grouped: defaultdict[tuple[int, str], defaultdict[Any, list[bool]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in table.to_pylist():
        subject = row.get("subject_id")
        prediction_time = row.get("prediction_time")
        if subject is None or prediction_time is None:
            raise ContractError(f"{task_name}: official label contains a null patient or prediction time.")
        split = split_map.get(int(subject))
        if split is None:
            raise ContractError(f"{task_name}: official label patient is missing from canonical splits.")
        if split not in allowed_splits:
            continue
        value = row.get(field)
        if value is None or not validate_domain(value, task):
            raise ContractError(f"{task_name}: train/validation label is null or outside its official domain.")
        grouped[(int(subject), split)][prediction_time].append(binary_target(value, task))
    result: dict[tuple[int, str], list[tuple[Any, bool]]] = {}
    for key, by_time in grouped.items():
        unique: list[tuple[Any, bool]] = []
        for prediction_time, outcomes in sorted(by_time.items()):
            if len(set(outcomes)) != 1:
                raise ContractError(
                    f"{task_name}: one official patient/time has conflicting binary targets; selection is ambiguous."
                )
            unique.append((prediction_time, outcomes[0]))
        result[key] = unique
    return result


def _aggregate_summary(rows: list[dict[str, Any]], spec: FinalM1Authorization) -> dict[str, Any]:
    counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        counter = counts[(str(row["task"]), str(row["split"]))]
        counter["instances"] += 1
        counter["positive_instances"] += int(bool(row["target"]))
    groups = []
    for task in spec.tasks:
        for split in spec.splits:
            counter = counts[(task, split)]
            total = int(counter["instances"])
            positive = int(counter["positive_instances"])
            if total < 1:
                raise ContractError(f"Final cohort has no {split} instances for {task}.")
            groups.append(
                {
                    "task": task,
                    "split": split,
                    "instances": total,
                    "positive_instances": positive,
                    "negative_instances": total - positive,
                    "prevalence": positive / total,
                }
            )
    return {
        "groups": groups,
        "selected_instances": len(rows),
        "maximum_instances_per_patient_task": 1,
        "test_instances_selected": 0,
        "test_predictions_or_performance_used": False,
    }


def build_final_cohort(
    *,
    asset_root: Path,
    registry: Mapping[str, Any],
    spec: FinalM1Authorization,
) -> dict[str, Any]:
    """Write or hash-verify the versioned final cohort without inspecting test outcomes."""
    output = spec.cohort_path
    complete_path = output.parent / "COMPLETE.json"
    config_hash = stable_hash(spec.raw)
    if complete_path.is_file():
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if (
            payload.get("state") != "COMPLETE"
            or payload.get("experiment_id") != spec.experiment_id
            or payload.get("experiment_config_hash") != config_hash
            or not output.is_file()
            or payload.get("cohort_sha256") != file_sha256(output)
        ):
            raise ContractError("Existing final cohort completion record is stale or hash-mismatched.")
        return dict(payload["aggregate_summary"])
    if output.exists():
        raise ContractError("A partial final cohort exists without an immutable COMPLETE record.")

    split_map = _split_map(asset_root, registry["split_mapping"])
    release = spec.raw["dataset_release"]
    release_id = str(release["release_id"])
    seed = int(spec.raw["selection"]["frozen_seed"])
    selected: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for task_name in spec.tasks:
        task = registry["tasks"][task_name]
        field = str(task["label_field"])
        suffix = f"labels/{task_name}/labels.parquet"
        table = read_parquet_member(asset_root, suffix, ["subject_id", "prediction_time", field])
        source_hashes[task_name] = _zip_member_sha256(asset_root, suffix)
        grouped = _validated_unique_times(
            table,
            task_name=task_name,
            task=task,
            split_map=split_map,
            allowed_splits=spec.splits,
        )
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
            instance_key = stable_hash(
                {
                    "experiment_id": spec.experiment_id,
                    "selection_hash": rank,
                    "official_prediction_time": prediction_time,
                }
            )
            selected.append(
                {
                    "instance_key": instance_key,
                    "selection_hash": rank,
                    "subject_id": subject_id,
                    "prediction_time": prediction_time,
                    "task": task_name,
                    "family": str(task["family"]),
                    "split": split,
                    "target": bool(target),
                    "dataset_release_id": release_id,
                    "selection_seed": seed,
                    "eligible_prediction_time_count": len(eligible),
                    "selected_prediction_time_index": index,
                }
            )
    selected.sort(key=lambda row: (row["task"], row["split"], row["instance_key"]))
    uniqueness = {(row["subject_id"], row["task"]) for row in selected}
    if len(uniqueness) != len(selected):
        raise ContractError("Final cohort contains more than one selected instance per patient/task.")
    if {row["task"] for row in selected} != set(spec.tasks) or {row["split"] for row in selected} != set(
        spec.splits
    ):
        raise ContractError("Final cohort does not exactly cover all locked tasks and modeling splits.")
    schema = pa.schema(
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
    _atomic_parquet(output, selected, schema)
    aggregate = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "dataset_release_id": release_id,
        "selection_seed": seed,
        "selection_policy": spec.raw["selection"]["policy"],
        "selection_uses_outcome": False,
        "selection_uses_history_or_record_length": False,
        "selection_uses_model_result": False,
        "source_label_artifact_hashes": source_hashes,
        **_aggregate_summary(selected, spec),
    }
    text = json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    complete = {
        "state": "COMPLETE",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": config_hash,
        "cohort_sha256": file_sha256(output),
        "cohort_rows": len(selected),
        "source_label_artifact_hashes": source_hashes,
        "aggregate_summary": aggregate,
    }
    atomic_json(complete_path, complete)
    return aggregate
