from __future__ import annotations

import duckdb
import pytest

from temporal_ehr.data.contracts import (
    ContractError,
    deterministic_order_clause,
    validate_required_null_counts,
    validate_schema,
)


def test_schema_validation_accepts_expected_event_contract() -> None:
    validate_schema(
        {
            "subject_id": "BIGINT",
            "time": "TIMESTAMP",
            "code": "VARCHAR",
            "numeric_value": "FLOAT",
            "text_value": "VARCHAR",
            "unit": "VARCHAR",
            "omop_table": "VARCHAR",
        }
    )


def test_schema_validation_rejects_missing_or_changed_fields() -> None:
    with pytest.raises(ContractError):
        validate_schema({"subject_id": "VARCHAR", "time": "DATE", "code": "VARCHAR"})


def test_null_key_rejection() -> None:
    with pytest.raises(ContractError, match="Required-key null"):
        validate_required_null_counts({"subject_id": 0, "time": 1, "code": 0})


def test_deterministic_order_preserves_timestamp_precision(synthetic_event_path) -> None:
    clause = deterministic_order_clause()
    assert '"time" NULLS FIRST' in clause
    assert "CAST" not in clause.upper()
    connection = duckdb.connect()
    try:
        first = connection.execute(
            f"SELECT * FROM read_parquet(?) ORDER BY {clause}", [str(synthetic_event_path)]
        ).fetchall()
        second = connection.execute(
            f"SELECT * FROM read_parquet(?) ORDER BY {clause}", [str(synthetic_event_path)]
        ).fetchall()
    finally:
        connection.close()
    assert first == second
