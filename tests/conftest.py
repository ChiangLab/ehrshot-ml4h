"""Entirely synthetic MEDS-style fixtures for Phase 0 tests."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def synthetic_event_path(tmp_path: Path) -> Path:
    """Create invented events that exercise recurrence, concurrency, and null handling."""
    rows = [
        (101, dt.datetime(2010, 1, 1), "Gender/Synthetic", None, None, None, "person"),
        (101, dt.datetime(2020, 1, 1, 9), "LOINC/SYN-A", 1.0, None, "arb", "measurement"),
        (101, dt.datetime(2020, 1, 1, 9), "LOINC/SYN-A", 1.0, None, "arb", "measurement"),
        (101, dt.datetime(2020, 1, 1, 10), "LOINC/SYN-A", 2.0, None, "arb", "measurement"),
        (101, dt.datetime(2021, 6, 1), "SNOMED/SYN-X", None, "invented", None, "observation"),
        (101, dt.datetime(2022, 12, 1), "LOINC/SYN-A", 3.0, None, "arb", "measurement"),
        (101, dt.datetime(2023, 1, 1), "RxNorm/SYN-R", None, None, None, "drug_exposure"),
        (202, dt.datetime(2011, 2, 1), "Gender/Synthetic", None, None, None, "person"),
        (202, dt.datetime(2019, 3, 3, 8), "Visit/IP", None, None, None, "visit_occurrence"),
        (202, dt.datetime(2019, 3, 3, 8), "SNOMED/SYN-Y", None, None, None, "condition_occurrence"),
        (202, dt.datetime(2022, 8, 17), "SNOMED/SYN-Y", None, None, None, "condition_occurrence"),
        (202, dt.datetime(2024, 1, 5), "SNOMED/SyntheticDeath", None, None, None, "death"),
        (303, dt.datetime(2012, 5, 5), "Gender/Synthetic", None, None, None, "person"),
        (303, dt.datetime(2020, 7, 7), "Visit/OP", None, None, None, "visit_occurrence"),
    ]
    columns = list(zip(*rows, strict=True))
    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64(), nullable=False),
            pa.field("time", pa.timestamp("us"), nullable=False),
            pa.field("code", pa.string(), nullable=False),
            pa.field("numeric_value", pa.float32()),
            pa.field("text_value", pa.string()),
            pa.field("unit", pa.string()),
            pa.field("omop_table", pa.string()),
        ]
    )
    table = pa.Table.from_arrays(
        [pa.array(values, type=field.type) for values, field in zip(columns, schema, strict=True)],
        schema=schema,
    )
    path = tmp_path / "entirely_synthetic_events.parquet"
    pq.write_table(table, path)
    return path
