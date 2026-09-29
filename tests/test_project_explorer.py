from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.project_explorer import build_project_explorer, embedding_preview
from temporal_ehr.project_explorer_web import ExplorerApplication


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _event_rows() -> list[dict]:
    prediction = datetime(2026, 1, 31, 12)
    before = {
        "subject_id": 1,
        "time": prediction - timedelta(days=5),
        "code": "LOINC/1",
        "numeric_value": np.float32(10),
        "text_value": None,
        "unit": "g/dL",
        "omop_table": "measurement",
    }
    return [
        before,
        dict(before),
        {
            "subject_id": 1,
            "time": prediction,
            "code": "FUTURE/EXCLUDED",
            "numeric_value": np.float32(1),
            "text_value": None,
            "unit": None,
            "omop_table": "measurement",
        },
        {
            "subject_id": 2,
            "time": prediction - timedelta(days=95),
            "code": "SNOMED/2",
            "numeric_value": None,
            "text_value": "present",
            "unit": None,
            "omop_table": "condition_occurrence",
        },
        {
            "subject_id": 99,
            "time": prediction - timedelta(days=1),
            "code": "NOT/SELECTED",
            "numeric_value": None,
            "text_value": None,
            "unit": None,
            "omop_table": "observation",
        },
    ]


def _cohort_rows(split: str = "train") -> list[dict]:
    prediction = datetime(2026, 1, 31, 12)
    return [
        {
            "instance_key": "a" * 64,
            "selection_hash": "b" * 64,
            "subject_id": 1,
            "prediction_time": prediction,
            "task": "lab_anemia",
            "family": "lab",
            "split": split,
            "target": True,
            "dataset_release_id": "synthetic",
            "selection_seed": 7,
            "eligible_prediction_time_count": 1,
            "selected_prediction_time_index": 0,
        },
        {
            "instance_key": "c" * 64,
            "selection_hash": "d" * 64,
            "subject_id": 2,
            "prediction_time": prediction,
            "task": "new_hypertension",
            "family": "incident_diagnosis",
            "split": "validation",
            "target": False,
            "dataset_release_id": "synthetic",
            "selection_seed": 7,
            "eligible_prediction_time_count": 2,
            "selected_prediction_time_index": 1,
        },
    ]


def _fixture_root(tmp_path: Path, *, split: str = "train") -> tuple[Path, Path]:
    root = tmp_path / "repository"
    root.mkdir()
    event_path = tmp_path / "data.parquet"
    _write_parquet(event_path, _event_rows())
    _write_parquet(
        root / "outputs/private/final/locked_four_task_m1_v1/cohort/final_panel_cohort_v1.parquet",
        _cohort_rows(split),
    )
    prepared = root / "outputs/private/final/locked_four_task_m1_v1/prepared/lab_anemia/texts.parquet"
    _write_parquet(
        prepared,
        [
            {
                "task": "lab_anemia",
                "split": split,
                "instance_key": "a" * 64,
                "target": True,
                "representation": "C1_correct_compact_time",
                "perturbation_seed": None,
                "representation_key": "r" * 64,
                "token_count": 12,
                "truncated": False,
                "text": "[TIME=1_30_DAYS_BEFORE_T]\nLOINC/1 value=10 unit=g/dL",
            }
        ],
    )
    embedding_root = (
        root
        / "outputs/private/final/locked_four_task_m1_v1/embeddings/primary/lab_anemia/c1"
    )
    embedding_root.mkdir(parents=True)
    vectors = np.asarray([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    np.save(embedding_root / "embeddings.npy", vectors)
    metadata = [
        {"task": "lab_anemia", "split": split, "instance_key": "a" * 64, "target": True,
         "condition_embedding_row_index": 0, "model_role": "primary"},
        {"task": "lab_anemia", "split": "validation", "instance_key": "e" * 64, "target": False,
         "condition_embedding_row_index": 1, "model_role": "primary"},
    ]
    _write_parquet(embedding_root / "patient_window_metadata.parquet", metadata)
    (embedding_root / "SUMMARY.json").write_text(
        json.dumps({"condition": "C1_correct_compact_time"}), encoding="utf-8"
    )
    (embedding_root / "COMPLETE.json").write_text(
        json.dumps({"embeddings_sha256": file_sha256(embedding_root / "embeddings.npy")}),
        encoding="utf-8",
    )
    return root, event_path


def test_build_explorer_is_validation_only_and_strictly_pre_t(tmp_path: Path) -> None:
    root, event_path = _fixture_root(tmp_path)
    database = root / "outputs/private/inspection/explorer.duckdb"
    report = build_project_explorer(
        repository_root=root,
        event_path=event_path,
        database_path=database,
        source_commit="f" * 40,
        deep_validate_embeddings=True,
    )
    assert report["held_out_content_indexed"] is False
    assert report["embedding_artifacts"] == 1
    assert report["embedding_row_mappings"] == 2
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute(
            "SELECT count(*) FROM clinical.original_events_pre_prediction WHERE instance_key = ?",
            ["a" * 64],
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM clinical.deduplicated_events_pre_prediction WHERE instance_key = ?",
            ["a" * 64],
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT count(*) FROM clinical.original_events_pre_prediction WHERE code = 'FUTURE/EXCLUDED'"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT prediction_relative_window FROM clinical.deduplicated_events_pre_prediction "
            "WHERE instance_key = ?",
            ["a" * 64],
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT status FROM catalog.method_interfaces WHERE interface = 'M4'"
        ).fetchone()[0] == "absent from protocol"
        artifact = connection.execute("SELECT artifact_id FROM catalog.embedding_artifacts").fetchone()[0]
        preview = embedding_preview(connection, artifact, 0)
        assert preview["norm"] == pytest.approx(1.0)
        assert preview["first_coordinates"] == pytest.approx([0.6, 0.8])


def test_explorer_rejects_test_split(tmp_path: Path) -> None:
    root, event_path = _fixture_root(tmp_path, split="test")
    with pytest.raises(ContractError, match="prohibited split"):
        build_project_explorer(
            repository_root=root,
            event_path=event_path,
            database_path=root / "outputs/private/inspection/explorer.duckdb",
            source_commit="f" * 40,
        )


def test_web_application_has_bounded_interactive_views(tmp_path: Path) -> None:
    root, event_path = _fixture_root(tmp_path)
    database = root / "outputs/private/inspection/explorer.duckdb"
    build_project_explorer(
        repository_root=root,
        event_path=event_path,
        database_path=database,
        source_commit="f" * 40,
    )
    app = ExplorerApplication(database)
    overview = app.query("/api/overview", {})
    assert overview["cohort_total"] == 2
    assert overview["embedding_rows"] == 2
    cohort = app.query("/api/cohort", {"task": ["lab_anemia"], "split": ["train"]})
    assert len(cohort["rows"]) == 1
    events = app.query("/api/events", {"instance_key": ["a" * 64]})
    assert len(events["rows"]) == 1
    with pytest.raises(ContractError, match="Only train/validation"):
        app.query("/api/cohort", {"task": ["lab_anemia"], "split": ["test"]})
    with pytest.raises(ContractError, match="Unknown explorer route"):
        app.query("/api/sql", {})
