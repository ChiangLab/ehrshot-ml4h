"""Governed cohort access and streaming strictly censored event histories."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa

from temporal_ehr.data.contracts import ContractError

MODELING_SPLITS = ("train", "validation")


@dataclass(frozen=True)
class HistoryInstance:
    """One private modeling instance and its strictly prior event rows."""

    instance_key: str
    subject_id: int
    task: str
    split: str
    target: bool
    prediction_time: dt.datetime
    events: tuple[dict[str, Any], ...]


def validate_requested_splits(
    splits: Sequence[str],
    *,
    allow_test: bool = False,
) -> tuple[str, ...]:
    """Fail closed unless a request stays within its explicitly authorized splits."""
    requested = tuple(dict.fromkeys(splits))
    allowed = (*MODELING_SPLITS, "test") if allow_test else MODELING_SPLITS
    disallowed = sorted(set(requested) - set(allowed))
    if disallowed:
        raise ContractError(f"Modeling is restricted to {allowed}; disallowed split request: {disallowed}.")
    if "test" in requested and (not allow_test or requested != ("test",)):
        raise ContractError("Held-out access must be an explicit test-only request.")
    if not requested:
        raise ContractError("At least one modeling split must be requested.")
    return requested


def validate_cohort_splits(cohort_path: Path) -> dict[str, int]:
    """Verify that no patient crosses canonical splits and return aggregate split sizes."""
    connection = duckdb.connect()
    try:
        crossing = connection.execute(
            """
            SELECT count(*)
            FROM (
                SELECT subject_id
                FROM read_parquet(?)
                GROUP BY subject_id
                HAVING count(DISTINCT split) > 1
            )
            """,
            [str(cohort_path)],
        ).fetchone()[0]
        unexpected = connection.execute(
            "SELECT DISTINCT split FROM read_parquet(?) WHERE split NOT IN ('train', 'validation', 'test')",
            [str(cohort_path)],
        ).fetchall()
        counts = connection.execute(
            "SELECT split, count(*) FROM read_parquet(?) GROUP BY split ORDER BY split",
            [str(cohort_path)],
        ).fetchall()
    finally:
        connection.close()
    if crossing:
        raise ContractError(f"Canonical split contract failed: {crossing} patients cross splits.")
    if unexpected:
        raise ContractError("Canonical split contract failed: unexpected split names are present.")
    return {str(split): int(count) for split, count in counts}


def _history_query(task_filter: str, split_filter: str) -> str:
    return f"""
        SELECT
            c.instance_key,
            c.subject_id,
            c.task,
            c.split,
            c.target,
            c.prediction_time,
            e.time AS event_time,
            e.code,
            e.numeric_value,
            e.text_value,
            e.unit,
            e.omop_table
        FROM read_parquet(?) c
        LEFT JOIN read_parquet(?) e
          ON e.subject_id = c.subject_id
         AND e.time < c.prediction_time
        WHERE c.task IN ({task_filter}) AND c.split IN ({split_filter})
        ORDER BY
            c.instance_key,
            e.time NULLS FIRST,
            e.code NULLS FIRST,
            e.numeric_value NULLS FIRST,
            e.text_value NULLS FIRST,
            e.unit NULLS FIRST,
            e.omop_table NULLS FIRST
    """


def iter_strict_histories(
    event_path: Path,
    cohort_path: Path,
    *,
    tasks: Sequence[str],
    splits: Sequence[str] = MODELING_SPLITS,
    batch_size: int = 65_536,
    allow_test: bool = False,
) -> Iterator[HistoryInstance]:
    """Stream one history at a time; never materialize the full event table in Python."""
    requested_splits = validate_requested_splits(splits, allow_test=allow_test)
    if not tasks:
        raise ContractError("At least one task must be requested.")
    validate_cohort_splits(cohort_path)
    task_filter = ", ".join("?" for _ in tasks)
    split_filter = ", ".join("?" for _ in requested_splits)
    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    reader = connection.execute(
        _history_query(task_filter, split_filter),
        [str(cohort_path), str(event_path), *tasks, *requested_splits],
    ).to_arrow_reader(batch_size=batch_size)
    current_key: str | None = None
    current_metadata: tuple[int, str, str, bool, dt.datetime] | None = None
    current_events: list[dict[str, Any]] = []
    try:
        for batch in reader:
            for row in batch.to_pylist():
                instance_key = str(row["instance_key"])
                if current_key is not None and instance_key != current_key:
                    assert current_metadata is not None
                    yield HistoryInstance(current_key, *current_metadata, tuple(current_events))
                    current_events = []
                if instance_key != current_key:
                    current_key = instance_key
                    current_metadata = (
                        int(row["subject_id"]),
                        str(row["task"]),
                        str(row["split"]),
                        bool(row["target"]),
                        row["prediction_time"],
                    )
                if row["event_time"] is not None:
                    current_events.append(
                        {
                            "subject_id": int(row["subject_id"]),
                            "time": row["event_time"],
                            "code": row["code"],
                            "numeric_value": row["numeric_value"],
                            "text_value": row["text_value"],
                            "unit": row["unit"],
                            "omop_table": row["omop_table"],
                        }
                    )
        if current_key is not None:
            assert current_metadata is not None
            yield HistoryInstance(current_key, *current_metadata, tuple(current_events))
    finally:
        reader.close()
        connection.close()


def strict_history_sql_predicate() -> str:
    """Expose the governed predicate for tests and provenance records."""
    return "event_time < prediction_time"


def table_from_instances(instances: Sequence[dict[str, Any]]) -> pa.Table:
    """Build an Arrow cohort table for synthetic tests only."""
    return pa.Table.from_pylist(list(instances))
