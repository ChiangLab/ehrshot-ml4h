"""CPU-only, aggregate-only audit over an EHRSHOT Parquet event table."""

from __future__ import annotations

import datetime as dt
import os
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow.parquet as pq

from temporal_ehr.config import logical_path
from temporal_ehr.data import queries
from temporal_ehr.data.contracts import (
    EXPECTED_COLUMNS,
    REQUIRED_COLUMNS,
    validate_required_null_counts,
    validate_schema,
)
from temporal_ehr.data.inventory import sha256_file
from temporal_ehr.data.plots import generate_qc_figures


def _json_value(value: Any) -> Any:
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat(sep=" ") if isinstance(value, dt.datetime) else value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _query_rows(connection: duckdb.DuckDBPyConnection, sql: str, data_path: Path) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, [str(data_path)] * sql.count("?"))
    names = [description[0] for description in cursor.description]
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor.fetchall()]


def _query_one(connection: duckdb.DuckDBPyConnection, sql: str, data_path: Path) -> dict[str, Any]:
    rows = _query_rows(connection, sql, data_path)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one aggregate row, received {len(rows)}")
    return rows[0]


def parquet_file_metadata(data_path: Path, logical: str) -> dict[str, Any]:
    """Collect Parquet footer and writer details without reading event rows."""
    parquet = pq.ParquetFile(data_path)
    metadata = parquet.metadata
    compression_by_column: dict[str, set[str]] = {}
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            compression_by_column.setdefault(column.path_in_schema, set()).add(column.compression)
    key_names: list[str] = []
    if metadata.metadata:
        key_names = sorted(key.decode("utf-8", errors="replace") for key in metadata.metadata)
    return {
        "logical_path": logical,
        "format": "Apache Parquet",
        "size_bytes": int(data_path.stat().st_size),
        "sha256": sha256_file(data_path),
        "writer_metadata": {
            "created_by": metadata.created_by,
            "format_version": metadata.format_version,
            "serialized_size_bytes": int(metadata.serialized_size),
            "key_value_metadata_keys": key_names,
        },
        "row_groups": int(metadata.num_row_groups),
        "footer_row_count": int(metadata.num_rows),
        "compression_codecs": sorted({codec for codecs in compression_by_column.values() for codec in codecs}),
        "compression_by_column": {column: sorted(codecs) for column, codecs in sorted(compression_by_column.items())},
    }


def _quantile_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    probabilities = np.asarray(queries.QUANTILE_PROBABILITIES)
    quantiles = np.quantile(array, probabilities, method="linear")
    labels = ("min", "p25", "median", "p75", "p90", "p99", "max")
    return {
        "subject_count": int(array.size),
        "quantiles": {label: float(value) for label, value in zip(labels, quantiles, strict=True)},
    }


def _peak_rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bytes_used = usage if sys.platform == "darwin" else usage * 1024.0
    return bytes_used / (1024.0 * 1024.0)


def _comparison(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
    table_counts: Sequence[Mapping[str, Any]],
    schema: Mapping[str, str],
    file_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {
        "sha256": (expected["sha256"], file_metadata["sha256"]),
        "row_groups": (expected["row_groups"], file_metadata["row_groups"]),
        "event_count": (expected["event_count"], observed["event_count"]),
        "subject_count": (expected["subject_count"], observed["distinct_subject_count"]),
        "code_count": (expected["code_count"], observed["distinct_code_count"]),
        "earliest_time": (expected["earliest_time"], observed["earliest_time"]),
        "latest_time": (expected["latest_time"], observed["latest_time"]),
    }
    checks = [
        {"metric": metric, "expected": wanted, "observed": got, "matches": wanted == got}
        for metric, (wanted, got) in metrics.items()
    ]
    schema_checks = [
        {
            "column": column,
            "expected_type": expected_type,
            "observed_type": schema.get(column),
            "matches": schema.get(column) == expected_type,
        }
        for column, expected_type in expected["expected_columns"].items()
    ]
    observed_tables = {str(row["omop_table"]): int(row["event_count"]) for row in table_counts}
    table_checks = [
        {
            "omop_table": table,
            "expected": int(count),
            "observed": observed_tables.get(table),
            "matches": observed_tables.get(table) == int(count),
        }
        for table, count in expected["omop_table_counts"].items()
    ]
    observed_visit_count = observed_tables.get("visit_occurrence")
    return {
        "checks": checks,
        "schema_checks": schema_checks,
        "omop_table_checks": table_checks,
        "visit_count_discrepancy": {
            "published_benchmark_visits": int(expected["published_visit_count"]),
            "visit_occurrence_rows_in_parquet": observed_visit_count,
            "difference_parquet_minus_published": (
                observed_visit_count - int(expected["published_visit_count"])
                if observed_visit_count is not None
                else None
            ),
            "interpretation": (
                "Counts are not forced to match. A published encounter count can differ from "
                "MEDS visit_occurrence event-row counts because of definitions, transformations, "
                "or release versions."
            ),
        },
    }


def classify_readiness(audit: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the prespecified Phase 0 readiness labels; multiple flags may apply."""
    expected = audit["expected_comparison"]
    critical_metrics = {"event_count", "subject_count", "code_count"}
    metric_ok = all(check["matches"] for check in expected["checks"] if check["metric"] in critical_metrics)
    schema_ok = all(check["matches"] for check in expected["schema_checks"])
    keys_ok = all(
        row["null_count"] == 0 for row in audit["dataset"]["missingness"] if row["column"] in REQUIRED_COLUMNS
    )
    event_valid = bool(metric_ok and schema_ok and keys_ok)
    status_by_asset = {entry["asset"]: entry["status"] for entry in inventory["assets"]}
    core_assets = (
        "Task-label files for all 15 EHRSHOT tasks",
        "Official prediction-time fields",
        "Canonical train/validation/test patient splits",
    )
    core_present = all(status_by_asset.get(name) == "present" for name in core_assets)
    benchmark_assets = (
        *core_assets,
        "Canonical few-shot samples",
        "Code/concept descriptions",
        "Released baseline features or results",
    )
    benchmark_missing = any(status_by_asset.get(name) != "present" for name in benchmark_assets)
    released_manifest_missing = status_by_asset.get("Released checksums or release manifest") != "present"

    flags: list[str] = []
    if not event_valid:
        flags.append("DATA_INVALID")
    else:
        if core_present:
            flags.append("READY_FOR_LABEL_VALIDATION")
        if benchmark_missing:
            flags.append("EVENTS_VALID_BENCHMARK_ASSETS_MISSING")
        if released_manifest_missing:
            flags.append("DATA_VERSION_UNRESOLVED")
    return {
        "flags": flags,
        "event_table_valid": event_valid,
        "core_labels_prediction_times_splits_present": core_present,
        "all_benchmark_assets_present": not benchmark_missing,
        "release_provenance_cryptographically_resolved": not released_manifest_missing,
        "decision_note": (
            "Label validation may begin because all task-label schemas, prediction-time fields, "
            "and the split artifact are present. Few-shot samples and released baseline artifacts "
            "remain missing, and no released manifest binds the separately obtained files."
            if event_valid and core_present
            else "Do not begin label validation until the failed event or core-asset checks are resolved."
        ),
    }


def run_data_audit(
    data_path: Path,
    asset_root: Path,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the full governed-data audit and create aggregate QC figures."""
    started = time.perf_counter()
    logical = logical_path(data_path, asset_root)
    file_metadata = parquet_file_metadata(data_path, logical)
    connection = duckdb.connect(database=":memory:")
    thread_count = max(1, min(8, os.cpu_count() or 1))
    connection.execute(f"PRAGMA threads={thread_count}")
    connection.execute("PRAGMA disable_progress_bar")
    try:
        describe_rows = _query_rows(connection, queries.describe_sql(), data_path)
        schema = {str(row["column_name"]): str(row["column_type"]) for row in describe_rows}
        validate_schema(schema, EXPECTED_COLUMNS)

        basic = _query_one(connection, queries.basic_stats_sql(), data_path)
        null_row = _query_one(connection, queries.null_counts_sql(), data_path)
        validate_required_null_counts({key: int(value) for key, value in null_row.items()})
        total_events = int(basic["event_count"])
        missingness = [
            {
                "column": column,
                "null_count": int(null_row[column]),
                "null_fraction": float(null_row[column]) / total_events,
            }
            for column in queries.EVENT_COLUMNS
        ]
        for numerator in ("events_with_both_values", "events_with_neither_value"):
            basic[f"{numerator}_fraction"] = float(basic[numerator]) / total_events

        table_counts = _query_rows(connection, queries.table_counts_sql(), data_path)
        vocabulary_counts = _query_rows(
            connection,
            queries.vocabulary_counts_sql(config["assets"]["vocabulary_prefixes"]),
            data_path,
        )

        subject_values = _query_rows(connection, queries.subject_aggregate_values_sql(), data_path)
        subject_distributions = {
            metric: _quantile_summary([float(row[metric]) for row in subject_values])
            for metric in (
                "events_per_subject",
                "distinct_codes_per_subject",
                "timeline_duration_years",
                "unique_event_dates",
                "max_same_timestamp_concurrency",
            )
        }

        integrity: dict[str, Any] = {}
        for query in (
            queries.exact_duplicate_sql,
            queries.subject_date_code_duplicate_sql,
            queries.recurrence_sql,
            queries.concurrency_sql,
            queries.subject_integrity_sql,
            queries.chronological_order_sql,
            queries.range_check_sql,
        ):
            integrity.update(_query_one(connection, query(), data_path))
        integrity["null_timestamps"] = int(null_row["time"])
        integrity["definitions"] = {
            "exact_duplicate_row": "All seven stored column values are identical, including nulls.",
            "subject_date_code_duplicate": (
                "More than one row shares subject, calendar date, and code; values/timestamps may differ."
            ),
            "repeated_code_across_dates": (
                "A subject-code pair appears on more than one calendar date; this is recurrence, not an error."
            ),
            "same_timestamp_concurrency": (
                "More than one row shares a subject and exact timestamp; rows need not be duplicates."
            ),
            "administrative_tables": ["person", "visit_occurrence", "visit_detail"],
        }

        visit = _query_one(connection, queries.visit_audit_sql(), data_path)
        visit_prefix_counts = _query_rows(connection, queries.visit_code_prefix_counts_sql(), data_path)
        visit.update(
            {
                "code_prefix_counts": visit_prefix_counts,
                "explicit_visit_id_column_present": "visit_id" in schema,
                "encounter_identifier_recoverable": False,
                "start_end_concepts_distinguishable_from_schema": False,
                "recommended_grouping_if_no_external_encounter_map": "calendar-day visit blocks",
                "interpretation": (
                    "visit_occurrence is represented as event rows with four public vocabulary-prefix "
                    "categories; only a minority use Visit/. This file has neither visit_id nor an end-time "
                    "field. Code-string hints are not a validated encounter boundary, so true "
                    "encounters cannot be reconstructed from this event table alone."
                ),
                "terminology": {
                    "true_encounter": "A source-defined visit linked by a reliable encounter identifier.",
                    "calendar_day_group": "A deterministic fallback grouping; not a true encounter.",
                    "identical_timestamp": "Concurrent recorded time; not necessarily one encounter.",
                    "exact_duplicate": "Identical values in all seven stored columns.",
                },
            }
        )

        figure_paths = generate_qc_figures(
            output_dir / "figures",
            subject_values,
            table_counts,
            vocabulary_counts,
            missingness,
        )
    finally:
        connection.close()

    expected_comparison = _comparison(config["dataset"], basic, table_counts, schema, file_metadata)
    elapsed = time.perf_counter() - started
    return {
        "report_kind": "EHRSHOT aggregate data audit",
        "scope": "Full event table; aggregate results only",
        "file": file_metadata,
        "schema": [{"column": column, "type": column_type} for column, column_type in schema.items()],
        "dataset": {**basic, "missingness": missingness},
        "omop_table_counts": table_counts,
        "vocabulary_prefix_counts": vocabulary_counts,
        "subject_distributions": subject_distributions,
        "temporal_integrity": integrity,
        "visit_audit": visit,
        "expected_comparison": expected_comparison,
        "figures": figure_paths,
        "execution": {
            "wall_time_seconds": round(elapsed, 3),
            "peak_resident_memory_mb": round(_peak_rss_mb(), 1),
            "duckdb_threads": thread_count,
            "persistent_database_created": False,
        },
        "privacy": {
            "aggregate_only": True,
            "subject_identifiers_written": False,
            "individual_rows_written": False,
            "subject_level_intermediate_written": False,
            "absolute_source_paths_written": False,
        },
    }
