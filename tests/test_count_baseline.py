from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.features.count_baseline import build_sparse_counts, fit_vocabulary, run_count_baseline


def _modeling_fixture(tmp_path: Path) -> tuple[Path, Path]:
    events: list[dict[str, object]] = []
    cohort: list[dict[str, object]] = []
    prediction_time = dt.datetime(2024, 1, 1)
    for index in range(48):
        subject_id = 10_000 + index
        split = "train" if index < 24 else "validation"
        target = index % 2 == 0
        events.extend(
            [
                {
                    "subject_id": subject_id,
                    "time": dt.datetime(2020, 1, 1),
                    "code": "SNOMED/POS" if target else "SNOMED/NEG",
                    "numeric_value": None,
                    "text_value": None,
                    "unit": None,
                    "omop_table": "condition_occurrence",
                },
                {
                    "subject_id": subject_id,
                    "time": dt.datetime(2021, 1, 1),
                    "code": "SNOMED/COMMON",
                    "numeric_value": None,
                    "text_value": None,
                    "unit": None,
                    "omop_table": "condition_occurrence",
                },
                {
                    "subject_id": subject_id,
                    "time": prediction_time,
                    "code": "SNOMED/LEAK",
                    "numeric_value": None,
                    "text_value": None,
                    "unit": None,
                    "omop_table": "condition_occurrence",
                },
            ]
        )
        cohort.append(
            {
                "instance_key": f"synthetic-{index:03d}",
                "subject_id": subject_id,
                "prediction_time": prediction_time,
                "task": "synthetic_task",
                "family": "synthetic",
                "split": split,
                "target": target,
            }
        )
    event_path = tmp_path / "synthetic_model_events.parquet"
    cohort_path = tmp_path / "synthetic_model_cohort.parquet"
    pq.write_table(pa.Table.from_pylist(events), event_path)
    pq.write_table(pa.Table.from_pylist(cohort), cohort_path)
    return event_path, cohort_path


def test_training_vocabulary_and_sparse_counts_are_strict_and_log_transformed(tmp_path: Path) -> None:
    event_path, cohort_path = _modeling_fixture(tmp_path)
    vocabulary = fit_vocabulary(event_path, cohort_path, task="synthetic_task")
    assert "SNOMED/LEAK" not in vocabulary
    validation = build_sparse_counts(
        event_path,
        cohort_path,
        task="synthetic_task",
        split="validation",
        vocabulary=vocabulary,
        batch_size=5,
    )
    assert validation.matrix.shape == (24, 3)
    assert validation.eligible_event_count == 48
    assert set(np.unique(validation.matrix.data)) == {np.log1p(1.0)}


def test_count_baseline_is_deterministic_and_records_every_candidate(tmp_path: Path) -> None:
    event_path, cohort_path = _modeling_fixture(tmp_path)
    private_root = tmp_path / "private"
    report_root = tmp_path / "reports"
    first = run_count_baseline(
        event_path,
        cohort_path,
        tasks=("synthetic_task",),
        private_root=private_root,
        report_root=report_root,
        c_grid=(0.1, 1.0),
        random_seed=7,
    )
    second = run_count_baseline(
        event_path,
        cohort_path,
        tasks=("synthetic_task",),
        private_root=private_root,
        report_root=report_root,
        c_grid=(0.1, 1.0),
        random_seed=7,
    )
    assert first == second
    task = first["tasks"]["synthetic_task"]
    assert [row["C"] for row in task["candidate_results"]] == [0.1, 1.0]
    assert task["selected_C"] == 0.1
    assert task["metrics"]["status"] == "REPORTED"
    saved = json.loads((report_root / "validation_results.json").read_text())
    assert saved["test_split_inspected"] is False
    assert (private_root / "synthetic_task" / "model.joblib").exists()
