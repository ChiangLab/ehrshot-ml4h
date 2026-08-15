from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.history import iter_strict_histories, validate_cohort_splits, validate_requested_splits


def _write_cohort(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_modeling_split_request_fails_closed_on_test() -> None:
    with pytest.raises(ContractError, match="disallowed split request"):
        validate_requested_splits(("train", "test"))


def test_patient_cannot_cross_canonical_splits(tmp_path: Path) -> None:
    path = tmp_path / "synthetic_crossing_cohort.parquet"
    common = {
        "subject_id": 101,
        "prediction_time": dt.datetime(2023, 1, 1),
        "task": "synthetic_task",
        "family": "synthetic",
        "target": False,
    }
    _write_cohort(
        path,
        [
            {**common, "instance_key": "a", "split": "train"},
            {**common, "instance_key": "b", "split": "validation"},
        ],
    )
    with pytest.raises(ContractError, match="cross splits"):
        validate_cohort_splits(path)


def test_streamed_history_excludes_exact_and_later_events(synthetic_event_path: Path, tmp_path: Path) -> None:
    path = tmp_path / "synthetic_cohort.parquet"
    _write_cohort(
        path,
        [
            {
                "instance_key": "synthetic-instance",
                "subject_id": 101,
                "prediction_time": dt.datetime(2020, 1, 1, 9),
                "task": "synthetic_task",
                "family": "synthetic",
                "split": "train",
                "target": False,
            }
        ],
    )
    histories = list(
        iter_strict_histories(
            synthetic_event_path,
            path,
            tasks=("synthetic_task",),
            splits=("train",),
            batch_size=2,
        )
    )
    assert len(histories) == 1
    assert [event["code"] for event in histories[0].events] == ["Gender/Synthetic"]
    assert all(event["time"] < histories[0].prediction_time for event in histories[0].events)
