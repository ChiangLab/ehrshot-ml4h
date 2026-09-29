"""Governed, validation-only DuckDB catalog for interactive project review.

The catalog is intentionally an index rather than a second source of truth.  It
copies small metadata and summary text, but keeps the original event parquet and
full embedding arrays immutable in their governed locations.  Persistent views
and row maps make those artifacts inspectable without duplicating them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, validate_schema
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

SCHEMA_VERSION = "governed_project_explorer_v1"
ALLOWED_SPLITS = {"train", "validation"}
TASKS = ("lab_anemia", "guo_readmission", "new_hypertension", "chexpert")
PRIVATE_ROOT = Path("outputs/private/inspection")
DEFAULT_DATABASE = PRIVATE_ROOT / "temporal_ehr_project_explorer.duckdb"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"Expected a JSON object: {path}")
    return value


def _insert_many(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    values = [tuple(_jsonable(row.get(column)) for column in columns) for row in rows]
    if not values:
        return 0
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    return len(values)


def _event_schema(connection: duckdb.DuckDBPyConnection, event_path: Path) -> dict[str, str]:
    rows = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({_sql_literal(event_path)})"
    ).fetchall()
    schema = {str(row[0]): str(row[1]) for row in rows}
    validate_schema(schema)
    return schema


def _require_development_splits(rows: Sequence[Mapping[str, Any]], context: str) -> None:
    splits = {str(row.get("split")) for row in rows}
    if not splits or not splits <= ALLOWED_SPLITS:
        raise ContractError(f"{context} contains a missing or prohibited split: {sorted(splits)}")


def _embedding_diagnostics(
    path: Path,
    *,
    deep_validate: bool,
    expected_rows: int | None,
) -> dict[str, Any]:
    vectors = np.load(path, mmap_mode="r")
    if vectors.ndim != 2 or not vectors.shape[0] or not vectors.shape[1]:
        raise ContractError(f"Embedding array is not a nonempty matrix: {path}")
    if expected_rows is not None and vectors.shape[0] != expected_rows:
        raise ContractError(f"Embedding row count does not match metadata: {path}")
    result: dict[str, Any] = {
        "rows": int(vectors.shape[0]),
        "dimensions": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "all_finite": None,
        "norm_min": None,
        "norm_max": None,
        "unit_normalized": None,
    }
    if not deep_validate:
        return result
    finite = True
    norm_min = math.inf
    norm_max = -math.inf
    for start in range(0, len(vectors), 4096):
        block = np.asarray(vectors[start : start + 4096], dtype=np.float64)
        finite = finite and bool(np.isfinite(block).all())
        norms = np.linalg.norm(block, axis=1)
        norm_min = min(norm_min, float(norms.min()))
        norm_max = max(norm_max, float(norms.max()))
    result.update(
        {
            "all_finite": finite,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "unit_normalized": finite
            and math.isclose(norm_min, 1.0, abs_tol=1e-5)
            and math.isclose(norm_max, 1.0, abs_tol=1e-5),
        }
    )
    return result


def _artifact_identity(path: Path, repository_root: Path) -> str:
    return stable_hash({"schema": SCHEMA_VERSION, "path": str(path.relative_to(repository_root))})


def _condition_from_path(path: Path) -> str:
    summary = path.parent / "SUMMARY.json"
    if summary.is_file():
        return str(_read_json(summary).get("condition", path.parent.name))
    return path.parent.name


def _metadata_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    payload = _read_json(path)
    rows = payload.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ContractError(f"Embedding metadata rows are invalid: {path}")
    return [dict(row) for row in rows]


def _locate_metadata(vector_path: Path, method: str) -> Path:
    candidates = (
        (vector_path.parent / "patient_window_metadata.parquet", vector_path.parent / "patient_metadata.parquet")
        if method == "M1"
        else (vector_path.parent / "metadata.json", vector_path.parent / "patient_metadata.parquet")
        if method == "M2"
        else (vector_path.parent / "sequence_metadata.parquet",)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ContractError(f"No row metadata accompanies embedding array: {vector_path}")


def _embedding_files(repository_root: Path) -> list[tuple[str, Path]]:
    roots = {
        "M1": repository_root / "outputs/private/final/locked_four_task_m1_v1/embeddings",
        "M2": repository_root / "outputs/private/final/m2_structured_v2_reference_v1/embeddings",
        "M3": repository_root / "outputs/private/final/locked_four_task_m2_m3_v1/m3/window_embeddings",
    }
    output: list[tuple[str, Path]] = []
    for method, root in roots.items():
        if root.is_dir():
            output.extend((method, path) for path in sorted(root.rglob("embeddings.npy")))
    return output


def _embedding_row_map(
    *,
    method: str,
    artifact_id: str,
    metadata: Sequence[Mapping[str, Any]],
    vector_rows: int,
) -> Iterable[dict[str, Any]]:
    if method != "M3":
        for fallback, row in enumerate(metadata):
            index = row.get("condition_embedding_row_index", row.get("embedding_row_index", fallback))
            yield {
                "artifact_id": artifact_id,
                "vector_row_index": int(index),
                "instance_key": str(row.get("instance_key", "")),
                "split": str(row.get("split", "")),
                "target": bool(row.get("target")),
                "sequence_position": None,
                "window_index": row.get("window_index"),
            }
        return
    covered = 0
    for row in sorted(metadata, key=lambda value: int(value["sequence_row_index"])):
        offset = int(row["window_embedding_offset"])
        count = int(row["window_count"])
        if offset != covered or count < 1:
            raise ContractError("M3 sequence metadata is not contiguous.")
        for position in range(count):
            yield {
                "artifact_id": artifact_id,
                "vector_row_index": offset + position,
                "instance_key": str(row.get("instance_key", "")),
                "split": str(row.get("split", "")),
                "target": bool(row.get("target")),
                "sequence_position": position,
                "window_index": None,
            }
        covered += count
    if covered != vector_rows:
        raise ContractError("M3 sequence metadata does not cover every embedding row.")


def _task_from_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    values = {str(row.get("task")) for row in rows if row.get("task") is not None}
    if len(values) == 1:
        return next(iter(values))
    for task in TASKS:
        if task in path.parts:
            return task
    raise ContractError(f"Cannot determine task for artifact: {path}")


def _create_core_tables(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("CREATE SCHEMA catalog")
    connection.execute("CREATE SCHEMA clinical")
    connection.execute("CREATE SCHEMA representations")
    connection.execute("CREATE SCHEMA results")
    connection.execute(
        """
        CREATE TABLE catalog.manifest (
          schema_version VARCHAR, built_at_utc TIMESTAMP, source_commit VARCHAR,
          event_source_path VARCHAR, event_source_bytes BIGINT, database_scope VARCHAR,
          held_out_content_indexed BOOLEAN, full_embedding_arrays_copied BOOLEAN
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE catalog.method_interfaces (
          interface VARCHAR, descriptive_name VARCHAR, status VARCHAR, input_representation VARCHAR,
          classifier VARCHAR, development_scope VARCHAR, notes VARCHAR
        )
        """
    )
    connection.execute(
        "CREATE TABLE catalog.source_columns (column_name VARCHAR, logical_type VARCHAR)"
    )
    connection.execute(
        """
        CREATE TABLE catalog.embedding_artifacts (
          artifact_id VARCHAR PRIMARY KEY, method VARCHAR, task VARCHAR, condition VARCHAR,
          model_role VARCHAR, vector_path VARCHAR, metadata_path VARCHAR, complete_path VARCHAR,
          vector_sha256 VARCHAR, rows BIGINT, metadata_rows BIGINT, dimensions BIGINT, dtype VARCHAR,
          all_finite BOOLEAN, norm_min DOUBLE, norm_max DOUBLE, unit_normalized BOOLEAN,
          integrity_status VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE representations.embedding_rows (
          artifact_id VARCHAR, vector_row_index BIGINT, instance_key VARCHAR, split VARCHAR,
          target BOOLEAN, sequence_position BIGINT, window_index BIGINT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE representations.m1_inputs (
          task VARCHAR, split VARCHAR, instance_key VARCHAR, target BOOLEAN,
          representation VARCHAR, perturbation_seed BIGINT, representation_key VARCHAR,
          token_count BIGINT, truncated BOOLEAN, serialized_text VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE representations.m2_summaries (
          generation_key VARCHAR PRIMARY KEY, task VARCHAR, condition VARCHAR,
          source_text VARCHAR, summary_text VARCHAR, validation_json VARCHAR,
          evidence_status VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE representations.m2_summary_instances (
          generation_key VARCHAR, task VARCHAR, condition VARCHAR, split VARCHAR,
          instance_key VARCHAR, target BOOLEAN
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE results.validation_metrics (
          method VARCHAR, task VARCHAR, scope VARCHAR, condition VARCHAR,
          metric VARCHAR, value DOUBLE, source_report VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE catalog.run_status (
          job_id VARCHAR, state VARCHAR, run_kind VARCHAR, git_commit VARCHAR,
          started_at VARCHAR, ended_at VARCHAR, exit_status BIGINT
        )
        """
    )


def _insert_methods(connection: duckdb.DuckDBPyConnection) -> None:
    rows = [
        {"interface": "B1", "descriptive_name": "latest-window baseline", "status": "complete",
         "input_representation": "most recent retained time window", "classifier": "penalized logistic regression",
         "development_scope": "train/validation", "notes": "separate baseline, not M1"},
        {"interface": "B2", "descriptive_name": "latest-code baseline", "status": "complete",
         "input_representation": "latest code occurrence", "classifier": "penalized logistic regression",
         "development_scope": "train/validation", "notes": "separate baseline, not M1"},
        {"interface": "B3", "descriptive_name": "count baseline", "status": "complete",
         "input_representation": "log1p code counts", "classifier": "penalized logistic regression",
         "development_scope": "train/validation", "notes": "sparse non-embedding baseline"},
        {"interface": "B4", "descriptive_name": "prevalence baseline", "status": "complete",
         "input_representation": "training prevalence", "classifier": "constant probability",
         "development_scope": "train/validation", "notes": "calibration/performance reference"},
        {"interface": "M1", "descriptive_name": "flat-history embedding", "status": "complete",
         "input_representation": "Qwen3-Embedding-0.6B over serialized history",
         "classifier": "task-specific L2-regularized logistic regression",
         "development_scope": "train/validation; one frozen held-out evaluation kept outside this explorer",
         "notes": "C0-C3 temporal falsification plus selected 8B confirmation"},
        {"interface": "M2", "descriptive_name": "structured-summary embedding", "status": "production pending",
         "input_representation": "on-prem Qwen3.6 structured summary, then Qwen3-Embedding-0.6B",
         "classifier": "task-specific L2-regularized logistic regression",
         "development_scope": "train/validation only", "notes": "manual fidelity gate precedes fitting"},
        {"interface": "M3", "descriptive_name": "window-sequence GRU", "status": "reference complete",
         "input_representation": "Qwen3-Embedding-0.6B for each time window",
         "classifier": "single-layer unidirectional GRU",
         "development_scope": "train/validation only", "notes": "three training seeds"},
        {"interface": "M4", "descriptive_name": "not defined", "status": "absent from protocol",
         "input_representation": "none", "classifier": "none", "development_scope": "none",
         "notes": "The governed repository defines M1-M3; no M4 result may be inferred."},
    ]
    _insert_many(connection, "catalog.method_interfaces", tuple(rows[0]), rows)


def _insert_m1_inputs(connection: duckdb.DuckDBPyConnection, repository_root: Path) -> int:
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        path = repository_root / f"outputs/private/final/locked_four_task_m1_v1/prepared/{task}/texts.parquet"
        if not path.is_file():
            continue
        for source in pq.read_table(path).to_pylist():
            if str(source.get("split")) not in ALLOWED_SPLITS:
                raise ContractError("M1 prepared inputs include a prohibited split.")
            rows.append(
                {
                    "task": task,
                    "split": source.get("split"),
                    "instance_key": source.get("instance_key"),
                    "target": source.get("target"),
                    "representation": source.get("representation"),
                    "perturbation_seed": source.get("perturbation_seed"),
                    "representation_key": source.get("representation_key"),
                    "token_count": source.get("token_count"),
                    "truncated": source.get("truncated"),
                    "serialized_text": source.get("text"),
                }
            )
    columns = (
        "task", "split", "instance_key", "target", "representation", "perturbation_seed",
        "representation_key", "token_count", "truncated", "serialized_text",
    )
    return _insert_many(connection, "representations.m1_inputs", columns, rows)


def _insert_embeddings(
    connection: duckdb.DuckDBPyConnection,
    repository_root: Path,
    *,
    deep_validate: bool,
) -> tuple[int, int]:
    artifact_count = 0
    row_count = 0
    artifact_columns = (
        "artifact_id", "method", "task", "condition", "model_role", "vector_path", "metadata_path",
        "complete_path", "vector_sha256", "rows", "metadata_rows", "dimensions", "dtype", "all_finite",
        "norm_min", "norm_max", "unit_normalized", "integrity_status",
    )
    row_columns = (
        "artifact_id", "vector_row_index", "instance_key", "split", "target", "sequence_position", "window_index",
    )
    for method, vector_path in _embedding_files(repository_root):
        metadata_path = _locate_metadata(vector_path, method)
        metadata = _metadata_rows(metadata_path)
        _require_development_splits(metadata, f"{method} embedding metadata")
        expected = None if method == "M3" else len(metadata)
        diagnostics = _embedding_diagnostics(vector_path, deep_validate=deep_validate, expected_rows=expected)
        complete_path = vector_path.parent / "COMPLETE.json"
        complete = _read_json(complete_path) if complete_path.is_file() else {}
        expected_hash = complete.get("embeddings_sha256")
        actual_hash = file_sha256(vector_path) if deep_validate or expected_hash else None
        if expected_hash and actual_hash != expected_hash:
            raise ContractError(f"Embedding array hash differs from completion record: {vector_path}")
        artifact_id = _artifact_identity(vector_path, repository_root)
        task = _task_from_rows(vector_path, metadata)
        model_role = str(metadata[0].get("model_role", method)) if metadata else method
        condition = _condition_from_path(vector_path)
        artifact = {
            "artifact_id": artifact_id,
            "method": method,
            "task": task,
            "condition": condition,
            "model_role": model_role,
            "vector_path": str(vector_path),
            "metadata_path": str(metadata_path),
            "complete_path": str(complete_path) if complete_path.is_file() else None,
            "vector_sha256": actual_hash or expected_hash,
            "rows": diagnostics["rows"],
            "metadata_rows": len(metadata),
            "dimensions": diagnostics["dimensions"],
            "dtype": diagnostics["dtype"],
            "all_finite": diagnostics["all_finite"],
            "norm_min": diagnostics["norm_min"],
            "norm_max": diagnostics["norm_max"],
            "unit_normalized": diagnostics["unit_normalized"],
            "integrity_status": "deep_verified" if deep_validate else "metadata_verified",
        }
        _insert_many(connection, "catalog.embedding_artifacts", artifact_columns, [artifact])
        mapped = _insert_many(
            connection,
            "representations.embedding_rows",
            row_columns,
            _embedding_row_map(
                method=method,
                artifact_id=artifact_id,
                metadata=metadata,
                vector_rows=int(diagnostics["rows"]),
            ),
        )
        if mapped != int(diagnostics["rows"]):
            raise ContractError(f"Embedding row map is incomplete: {vector_path}")
        artifact_count += 1
        row_count += mapped
    return artifact_count, row_count


def _insert_m2_summaries(connection: duckdb.DuckDBPyConnection, repository_root: Path) -> tuple[int, int]:
    private = repository_root / "outputs/private/final/m2_structured_v2_reference_v1"
    plan_path = private / "PLAN.json"
    if not plan_path.is_file():
        return 0, 0
    summary_rows: dict[str, dict[str, Any]] = {}
    instance_rows: list[dict[str, Any]] = []
    for task in TASKS:
        task_path = private / "plan" / f"{task}.json"
        if not task_path.is_file():
            continue
        rows = _read_json(task_path).get("rows", [])
        if not isinstance(rows, list):
            raise ContractError("M2 frozen plan rows are invalid.")
        for row in rows:
            if str(row.get("split")) not in ALLOWED_SPLITS:
                raise ContractError("M2 production plan contains a prohibited split.")
            key = str(row["generation_key"])
            assessment_path = private / "generations" / key / "assessment.json"
            source_path = private / "generations" / key / "source.json"
            if assessment_path.is_file() and source_path.is_file():
                assessment = _read_json(assessment_path)
                source = _read_json(source_path)
                summary_rows.setdefault(
                    key,
                    {
                        "generation_key": key,
                        "task": task,
                        "condition": row.get("condition"),
                        "source_text": source.get("source"),
                        "summary_text": assessment.get("summary"),
                        "validation_json": json.dumps(assessment.get("validation", {}), sort_keys=True),
                        "evidence_status": assessment.get("status"),
                    },
                )
            instance_rows.append(
                {
                    "generation_key": key,
                    "task": task,
                    "condition": row.get("condition"),
                    "split": row.get("split"),
                    "instance_key": row.get("instance_key"),
                    "target": row.get("target"),
                }
            )
    summary_columns = (
        "generation_key", "task", "condition", "source_text", "summary_text", "validation_json", "evidence_status",
    )
    instance_columns = ("generation_key", "task", "condition", "split", "instance_key", "target")
    return (
        _insert_many(connection, "representations.m2_summaries", summary_columns, summary_rows.values()),
        _insert_many(connection, "representations.m2_summary_instances", instance_columns, instance_rows),
    )


def _metric_rows(repository_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(method: str, task: str, scope: str, condition: str, metrics: Mapping[str, Any], source: Path) -> None:
        for metric in ("auroc", "auprc", "brier_score", "log_loss"):
            if metric in metrics:
                rows.append({"method": method, "task": task, "scope": scope, "condition": condition,
                             "metric": metric, "value": float(metrics[metric]),
                             "source_report": str(source.relative_to(repository_root))})

    m1_path = repository_root / "reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json"
    if m1_path.is_file():
        payload = _read_json(m1_path)
        if payload.get("test_predictions_or_performance_used") is not False:
            raise ContractError("M1 development report does not preserve the test boundary.")
        for task, task_result in payload.get("tasks", {}).items():
            for scope, scope_result in task_result.get("primary_scopes", {}).items():
                for condition, result in scope_result.get("representations", {}).items():
                    method = condition.split("_", 1)[0] if condition.startswith(("B1_", "B2_")) else "M1"
                    add(method, task, scope, condition, result.get("metrics", {}), m1_path)
            b3 = task_result.get("B3_code_count", {}).get("metrics", {}).get("model", {})
            add("B3", task, "complete_cohort", "B3_code_count", b3, m1_path)
            add("B4", task, "complete_cohort", "B4_training_prevalence",
                task_result.get("B4_training_prevalence", {}).get("metrics", {}), m1_path)

    m3_path = repository_root / "reports/generated/final/locked_four_task_m2_m3_v1/m3_reference.json"
    if m3_path.is_file():
        payload = _read_json(m3_path)
        if payload.get("test_predictions_or_performance_used") is not False:
            raise ContractError("M3 report does not preserve the test boundary.")
        for task, task_result in payload.get("tasks", {}).items():
            for condition, result in task_result.get("conditions", {}).items():
                add("M3", task, "reference", condition, result.get("averaged_seed_metrics", {}), m3_path)
            add("M3_mean_pool", task, "reference", "mean_pool_logistic_baseline",
                task_result.get("mean_pool_logistic_baseline", {}).get("metrics", {}), m3_path)

    m2_path = repository_root / "reports/generated/final/m2_structured_v2_reference_v1/reference_validation.json"
    if m2_path.is_file():
        payload = _read_json(m2_path)
        if payload.get("test_data_used") is not False:
            raise ContractError("M2 report does not preserve the test boundary.")
        for task, task_result in payload.get("tasks", {}).items():
            for condition, result in task_result.get("representations", {}).items():
                add("M2", task, "reference", condition, result.get("metrics", {}), m2_path)
    return rows


def _insert_run_status(connection: duckdb.DuckDBPyConnection, repository_root: Path) -> int:
    rows = []
    status_root = repository_root / "outputs/private/run_status"
    if not status_root.is_dir():
        return 0
    prohibited = ("submission", "held_out", "one_time_test", "test_")
    for directory in sorted(path for path in status_root.iterdir() if path.is_dir()):
        states = [state for state in ("COMPLETE", "FAILED", "RUNNING") if (directory / f"{state}.json").is_file()]
        if not states:
            continue
        state = "FAILED" if "FAILED" in states else "COMPLETE" if "COMPLETE" in states else "RUNNING"
        payload = _read_json(directory / f"{state}.json")
        run_kind = str(payload.get("run_kind", ""))
        if any(token in run_kind.lower() for token in prohibited):
            continue
        rows.append(
            {
                "job_id": directory.name,
                "state": state,
                "run_kind": run_kind,
                "git_commit": payload.get("git_commit"),
                "started_at": payload.get("started_at"),
                "ended_at": payload.get("ended_at"),
                "exit_status": payload.get("exit_status"),
            }
        )
    columns = ("job_id", "state", "run_kind", "git_commit", "started_at", "ended_at", "exit_status")
    return _insert_many(connection, "catalog.run_status", columns, rows)


def build_project_explorer(
    *,
    repository_root: Path,
    event_path: Path,
    database_path: Path,
    source_commit: str,
    deep_validate_embeddings: bool = False,
) -> dict[str, Any]:
    """Atomically build a train/validation-only governed project explorer."""
    root = repository_root.resolve()
    event_path = event_path.resolve()
    database_path = database_path.resolve()
    cohort_path = root / "outputs/private/final/locked_four_task_m1_v1/cohort/final_panel_cohort_v1.parquet"
    if not event_path.is_file() or not cohort_path.is_file():
        raise ContractError("The original EHRSHOT parquet and final development cohort are required.")
    if len(source_commit) != 40 or any(character not in "0123456789abcdef" for character in source_commit):
        raise ContractError("Explorer build requires the full Git source commit.")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{database_path.name}.", suffix=".duckdb", dir=database_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        connection = duckdb.connect(str(temporary))
        try:
            source_schema = _event_schema(connection, event_path)
            _create_core_tables(connection)
            _insert_many(
                connection,
                "catalog.source_columns",
                ("column_name", "logical_type"),
                (
                    {"column_name": name, "logical_type": logical_type}
                    for name, logical_type in source_schema.items()
                ),
            )
            _insert_methods(connection)
            cohort_literal = _sql_literal(cohort_path)
            event_literal = _sql_literal(event_path)
            connection.execute(
                f"CREATE TABLE clinical.cohort_instances AS SELECT * FROM read_parquet({cohort_literal})"
            )
            bad_splits = connection.execute(
                "SELECT DISTINCT split FROM clinical.cohort_instances WHERE split NOT IN ('train', 'validation')"
            ).fetchall()
            if bad_splits:
                raise ContractError(f"Development cohort includes prohibited splits: {bad_splits}")
            connection.execute(
                f"""
                CREATE VIEW clinical.original_events_pre_prediction AS
                SELECT c.instance_key, c.task, c.family, c.split, c.target, c.subject_id,
                       c.prediction_time, e.time AS event_time, e.code, e.numeric_value,
                       e.text_value, e.unit, e.omop_table,
                       date_diff('day', e.time, c.prediction_time) AS days_before_prediction,
                       CASE WHEN c.task = 'new_hypertension' THEN 90 ELSE 30 END AS window_width_days,
                       CAST(ceil(date_diff('second', e.time, c.prediction_time) /
                            (CASE WHEN c.task = 'new_hypertension' THEN 7776000.0 ELSE 2592000.0 END)) AS BIGINT)
                            AS prediction_relative_window
                FROM clinical.cohort_instances c
                JOIN read_parquet({event_literal}) e ON e.subject_id = c.subject_id
                WHERE e.time < c.prediction_time
                """
            )
            connection.execute(
                """
                CREATE VIEW clinical.deduplicated_events_pre_prediction AS
                SELECT DISTINCT instance_key, task, family, split, target, subject_id,
                       prediction_time, event_time, code, numeric_value, text_value, unit,
                       omop_table, days_before_prediction, window_width_days,
                       prediction_relative_window
                FROM clinical.original_events_pre_prediction
                """
            )
            connection.execute(
                """
                CREATE VIEW clinical.cohort_summary AS
                SELECT task, split, count(*) AS instances,
                       sum(CASE WHEN target THEN 1 ELSE 0 END) AS positive_instances,
                       avg(CAST(target AS DOUBLE)) AS prevalence
                FROM clinical.cohort_instances GROUP BY task, split ORDER BY task, split
                """
            )
            input_rows = _insert_m1_inputs(connection, root)
            artifacts, embedding_rows = _insert_embeddings(
                connection, root, deep_validate=deep_validate_embeddings
            )
            summaries, summary_instances = _insert_m2_summaries(connection, root)
            m2_report = root / "reports/generated/final/m2_structured_v2_reference_v1/reference_validation.json"
            if m2_report.is_file():
                connection.execute(
                    "UPDATE catalog.method_interfaces SET status = 'reference complete' WHERE interface = 'M2'"
                )
            elif summaries:
                connection.execute(
                    "UPDATE catalog.method_interfaces SET status = 'generation in progress' WHERE interface = 'M2'"
                )
            metric_rows = _metric_rows(root)
            metric_columns = ("method", "task", "scope", "condition", "metric", "value", "source_report")
            _insert_many(connection, "results.validation_metrics", metric_columns, metric_rows)
            runs = _insert_run_status(connection, root)
            connection.execute(
                "INSERT INTO catalog.manifest VALUES (?, current_timestamp, ?, ?, ?, ?, ?, ?)",
                [SCHEMA_VERSION, source_commit, str(event_path), event_path.stat().st_size,
                 "train_validation_only", False, False],
            )
            connection.execute("CREATE INDEX cohort_instance_key ON clinical.cohort_instances(instance_key)")
            connection.execute("CREATE INDEX input_instance_key ON representations.m1_inputs(instance_key)")
            connection.execute(
                "CREATE INDEX embedding_artifact_row "
                "ON representations.embedding_rows(artifact_id, vector_row_index)"
            )
            connection.execute(
                "CREATE INDEX summary_generation_key "
                "ON representations.m2_summary_instances(generation_key)"
            )
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        temporary.replace(database_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "database_path": str(database_path),
        "database_bytes": database_path.stat().st_size,
        "source_commit": source_commit,
        "scope": "train_validation_only",
        "held_out_content_indexed": False,
        "full_embedding_arrays_copied": False,
        "m1_input_rows": input_rows,
        "embedding_artifacts": artifacts,
        "embedding_row_mappings": embedding_rows,
        "m2_summaries_available": summaries,
        "m2_summary_instance_mappings": summary_instances,
        "validation_metric_rows": len(metric_rows),
        "run_status_rows": runs,
        "deep_embedding_validation": deep_validate_embeddings,
    }


def embedding_preview(
    connection: duckdb.DuckDBPyConnection,
    artifact_id: str,
    row_index: int,
    *,
    coordinates: int = 24,
) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT vector_path, rows, dimensions, method, task, condition
        FROM catalog.embedding_artifacts WHERE artifact_id = ?
        """,
        [artifact_id],
    ).fetchone()
    if row is None:
        raise ContractError("Unknown embedding artifact.")
    path, rows, dimensions, method, task, condition = row
    if not 0 <= row_index < int(rows):
        raise ContractError("Embedding row index is outside the verified artifact.")
    vectors = np.load(str(path), mmap_mode="r")
    vector = np.asarray(vectors[row_index], dtype=np.float64)
    metadata = connection.execute(
        """
        SELECT instance_key, split, target, sequence_position, window_index
        FROM representations.embedding_rows
        WHERE artifact_id = ? AND vector_row_index = ?
        """,
        [artifact_id, row_index],
    ).fetchone()
    return {
        "artifact_id": artifact_id,
        "method": method,
        "task": task,
        "condition": condition,
        "row_index": row_index,
        "dimensions": int(dimensions),
        "norm": float(np.linalg.norm(vector)),
        "minimum": float(vector.min()),
        "maximum": float(vector.max()),
        "mean": float(vector.mean()),
        "first_coordinates": [float(value) for value in vector[:coordinates]],
        "instance_key": metadata[0] if metadata else None,
        "split": metadata[1] if metadata else None,
        "target": metadata[2] if metadata else None,
        "sequence_position": metadata[3] if metadata else None,
        "window_index": metadata[4] if metadata else None,
    }


def database_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
