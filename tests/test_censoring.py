from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.data.censoring import validate_strict_censoring


def test_strict_censoring_excludes_exact_timestamp(synthetic_event_path: Path, tmp_path: Path) -> None:
    cohort = pa.Table.from_pylist(
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
        ]
    )
    cohort_path = tmp_path / "synthetic_cohort.parquet"
    pq.write_table(cohort, cohort_path)
    report = validate_strict_censoring(synthetic_event_path, cohort_path)
    group = report["aggregate_groups"][0]
    assert report["status"] == "PASS"
    assert group["median_eligible_events"] == 1
    assert group["exact_time_event_rows_excluded"] == 2
