"""Deterministic, label-independent selection of one real EHRSHOT instance per patient and task."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.task_registry import binary_target, validate_domain
from temporal_ehr.data.zip_assets import read_parquet_member


def _selection_digest(seed: int, task: str, subject_id: int, prediction_time: Any, source_row: int) -> str:
    material = f"{seed}|{task}|{subject_id}|{prediction_time.isoformat()}|{source_row}".encode()
    return hashlib.sha256(material).hexdigest()


def _split_map(asset_root: Path, split_mapping: Mapping[str, str]) -> dict[int, str]:
    table = read_parquet_member(asset_root, "metadata/subject_splits.parquet", ["subject_id", "split"])
    result: dict[int, str] = {}
    for subject_id, split in zip(table["subject_id"].to_pylist(), table["split"].to_pylist(), strict=True):
        if int(subject_id) in result:
            raise ContractError("Canonical split artifact contains a duplicate subject.")
        result[int(subject_id)] = str(split_mapping[split])
    return result


def build_patient_balanced_cohort(
    asset_root: Path,
    registry: Mapping[str, Any],
    tasks: Sequence[str],
    seed: int,
    private_output: Path,
) -> dict[str, Any]:
    """Write a governed cohort and return only privacy-safe aggregate metadata."""
    unknown = sorted(set(tasks) - set(registry["tasks"]))
    if unknown:
        raise ValueError(f"Unknown task names: {unknown}")
    split_map = _split_map(asset_root, registry["split_mapping"])
    selected_rows: list[dict[str, Any]] = []

    for task_name in tasks:
        task = registry["tasks"][task_name]
        field = str(task["label_field"])
        table = read_parquet_member(
            asset_root,
            f"labels/{task_name}/labels.parquet",
            ["subject_id", "prediction_time", field],
        )
        candidates: dict[int, tuple[str, dict[str, Any]]] = {}
        for source_row, row in enumerate(table.to_pylist()):
            subject_id = int(row["subject_id"])
            prediction_time = row["prediction_time"]
            value = row[field]
            if prediction_time is None or value is None or subject_id not in split_map:
                continue
            if not validate_domain(value, task):
                raise ContractError(f"{task_name}: source label outside its declared domain.")
            digest = _selection_digest(seed, task_name, subject_id, prediction_time, source_row)
            candidate = {
                "instance_key": digest,
                "subject_id": subject_id,
                "prediction_time": prediction_time,
                "task": task_name,
                "family": str(task["family"]),
                "split": split_map[subject_id],
                "target": binary_target(value, task),
            }
            current = candidates.get(subject_id)
            if current is None or digest < current[0]:
                candidates[subject_id] = (digest, candidate)
        selected_rows.extend(candidate for _, candidate in candidates.values())

    selected_rows.sort(key=lambda row: (row["task"], row["split"], row["instance_key"]))
    if not selected_rows:
        raise ContractError("No valid cohort instances were selected.")
    schema = pa.schema(
        [
            pa.field("instance_key", pa.string(), nullable=False),
            pa.field("subject_id", pa.int64(), nullable=False),
            pa.field("prediction_time", pa.timestamp("us"), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("family", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("target", pa.bool_(), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(selected_rows, schema=schema)
    private_output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, private_output, compression="zstd")

    counts: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    patient_counts: defaultdict[str, set[int]] = defaultdict(set)
    for row in selected_rows:
        key = (row["task"], row["split"])
        counts[key]["rows"] += 1
        counts[key]["positive_rows"] += int(row["target"])
        patient_counts[row["task"]].add(row["subject_id"])

    task_summary: dict[str, Any] = {}
    for task_name in tasks:
        splits: dict[str, Any] = {}
        for split in ("train", "validation", "test"):
            counter = counts[(task_name, split)]
            rows = counter["rows"]
            positives = counter["positive_rows"]
            splits[split] = {
                "rows": rows,
                "positive_rows": positives,
                "negative_rows": rows - positives,
                "prevalence": positives / rows if rows else None,
            }
        task_summary[task_name] = {
            "selected_patients": len(patient_counts[task_name]),
            "splits": splits,
        }
    return {
        "status": "PASS",
        "selection_policy": (
            "one source label row per patient and task; minimum seeded SHA-256 rank over task, patient, "
            "prediction time, and stable source-row position"
        ),
        "selection_uses_outcome": False,
        "selection_seed": seed,
        "private_output_logical_name": private_output.name,
        "selected_instance_count": len(selected_rows),
        "tasks": task_summary,
    }


def render_cohort_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Patient-balanced EHRSHOT cohort",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Selection: {report['selection_policy']}.",
        "The selection rank does not use the outcome. Canonical EHRSHOT patient splits are preserved.",
        "",
        "| Task | Split | Rows | Positive | Negative | Prevalence |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task_name, task in report["tasks"].items():
        for split, values in task["splits"].items():
            prevalence = values["prevalence"]
            prevalence_text = "NA" if prevalence is None else f"{prevalence:.3f}"
            lines.append(
                f"| {task_name} | {split} | {values['rows']:,} | {values['positive_rows']:,} | "
                f"{values['negative_rows']:,} | {prevalence_text} |"
            )
    lines.extend(
        [
            "",
            "The patient-level Parquet is a governed derivative and is excluded from Git. Do not upload it ",
            "to public GitHub or send it to an external model API.",
            "",
        ]
    )
    return "\n".join(lines)


def write_cohort_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2) + "\n"
    markdown = render_cohort_report(report)
    assert_privacy_safe_report(json_text)
    assert_privacy_safe_report(markdown)
    (output_dir / "cohort_summary.json").write_text(json_text, encoding="utf-8")
    (output_dir / "cohort_summary.md").write_text(markdown, encoding="utf-8")


def summarize_existing_cohort(cohort_path: Path, tasks: Sequence[str]) -> dict[str, Any]:
    """Return aggregate counts for an existing governed cohort without exposing rows."""
    connection = duckdb.connect()
    try:
        unknown_tasks = connection.execute(
            "SELECT DISTINCT task FROM read_parquet(?) WHERE task NOT IN (SELECT unnest(?)) ORDER BY task",
            [str(cohort_path), list(tasks)],
        ).fetchall()
        if unknown_tasks:
            raise ContractError("The governed cohort contains tasks outside the requested primary panel.")
        rows = connection.execute(
            """
            SELECT task, split, count(*) AS rows,
                   count(*) FILTER (WHERE target) AS positive_rows,
                   count(DISTINCT subject_id) AS patients
            FROM read_parquet(?)
            WHERE task IN (SELECT unnest(?))
            GROUP BY task, split
            ORDER BY task, split
            """,
            [str(cohort_path), list(tasks)],
        ).fetchall()
    finally:
        connection.close()
    groups = [
        {
            "task": task,
            "split": split,
            "rows": count,
            "positive_rows": positives,
            "negative_rows": count - positives,
            "patients": patients,
            "prevalence": positives / count if count else None,
        }
        for task, split, count, positives, patients in rows
    ]
    return {"status": "PASS", "groups": groups}
