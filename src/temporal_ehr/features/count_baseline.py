"""Streaming sparse features for a simple count-based reference."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.history import validate_cohort_splits, validate_requested_splits
from temporal_ehr.evaluation.metrics import validation_metrics

DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class SparseFeatureSet:
    """Private matrix plus aligned instance metadata."""

    matrix: sparse.csr_matrix
    instance_keys: tuple[str, ...]
    targets: np.ndarray
    split: str
    eligible_event_count: int


def fit_vocabulary(
    event_path: Path,
    cohort_path: Path,
    *,
    task: str,
    min_count: int = 1,
) -> dict[str, int]:
    """Fit a deterministic code vocabulary using strictly censored training events only."""
    validate_requested_splits(("train",))
    validate_cohort_splits(cohort_path)
    if min_count < 1:
        raise ValueError("min_count must be at least one.")
    connection = duckdb.connect()
    try:
        rows = connection.execute(
            """
            SELECT e.code, count(*) AS event_count
            FROM read_parquet(?) c
            JOIN read_parquet(?) e
              ON e.subject_id = c.subject_id
             AND e.time < c.prediction_time
            WHERE c.task = ? AND c.split = 'train'
            GROUP BY e.code
            HAVING count(*) >= ?
            ORDER BY e.code
            """,
            [str(cohort_path), str(event_path), task, min_count],
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ContractError(f"No strictly prior training events are available for {task}.")
    return {str(code): index for index, (code, _) in enumerate(rows)}


def build_sparse_counts(
    event_path: Path,
    cohort_path: Path,
    *,
    task: str,
    split: str,
    vocabulary: Mapping[str, int],
    batch_size: int = 65_536,
) -> SparseFeatureSet:
    """Stream grouped counts from DuckDB into CSR and apply log1p to nonzeros."""
    validate_requested_splits((split,))
    validate_cohort_splits(cohort_path)
    if not vocabulary:
        raise ContractError("The training vocabulary is empty.")
    expected_indices = set(range(len(vocabulary)))
    if set(vocabulary.values()) != expected_indices:
        raise ContractError("Vocabulary indices must be contiguous and zero-based.")

    connection = duckdb.connect()
    connection.execute("SET preserve_insertion_order = false")
    try:
        cohort_rows = connection.execute(
            """
            SELECT instance_key, target
            FROM read_parquet(?)
            WHERE task = ? AND split = ?
            ORDER BY instance_key
            """,
            [str(cohort_path), task, split],
        ).fetchall()
        if not cohort_rows:
            raise ContractError(f"No {split} instances are available for {task}.")
        instance_keys = tuple(str(row[0]) for row in cohort_rows)
        targets = np.asarray([bool(row[1]) for row in cohort_rows], dtype=np.int8)
        row_lookup = {key: index for index, key in enumerate(instance_keys)}

        vocabulary_table = pa.table(
            {
                "code": pa.array(list(vocabulary.keys()), type=pa.string()),
                "feature_index": pa.array(list(vocabulary.values()), type=pa.int32()),
            }
        )
        connection.register("training_vocabulary", vocabulary_table)
        reader = connection.execute(
            """
            SELECT
                c.instance_key,
                v.feature_index,
                count(*) AS event_count,
                bool_or(e.time >= c.prediction_time) AS leakage_detected
            FROM read_parquet(?) c
            JOIN read_parquet(?) e
              ON e.subject_id = c.subject_id
             AND e.time < c.prediction_time
            JOIN training_vocabulary v ON v.code = e.code
            WHERE c.task = ? AND c.split = ?
            GROUP BY c.instance_key, v.feature_index
            ORDER BY c.instance_key, v.feature_index
            """,
            [str(cohort_path), str(event_path), task, split],
        ).fetch_record_batch(rows_per_batch=batch_size)
        row_parts: list[np.ndarray] = []
        column_parts: list[np.ndarray] = []
        count_parts: list[np.ndarray] = []
        eligible_event_count = 0
        for batch in reader:
            payload = batch.to_pydict()
            if any(payload["leakage_detected"]):
                raise ContractError("An event at or after prediction time entered the feature query.")
            row_parts.append(np.fromiter((row_lookup[str(key)] for key in payload["instance_key"]), dtype=np.int64))
            column_parts.append(np.asarray(payload["feature_index"], dtype=np.int32))
            raw_counts = np.asarray(payload["event_count"], dtype=np.float64)
            eligible_event_count += int(raw_counts.sum())
            count_parts.append(np.log1p(raw_counts))
        reader.close()
    finally:
        connection.close()

    row_indices = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    column_indices = np.concatenate(column_parts) if column_parts else np.empty(0, dtype=np.int32)
    values = np.concatenate(count_parts) if count_parts else np.empty(0, dtype=np.float64)
    matrix = sparse.coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(instance_keys), len(vocabulary)),
        dtype=np.float64,
    ).tocsr()
    return SparseFeatureSet(matrix, instance_keys, targets, split, eligible_event_count)


def _fit_candidates(
    train: SparseFeatureSet,
    validation: SparseFeatureSet,
    *,
    c_grid: Sequence[float],
    random_seed: int,
) -> tuple[LogisticRegression, list[dict[str, float]], float]:
    if len(set(train.targets.tolist())) != 2 or len(set(validation.targets.tolist())) != 2:
        raise ContractError("Both train and validation must contain positive and negative outcomes.")
    records: list[dict[str, float]] = []
    models: dict[float, LogisticRegression] = {}
    for c_value in c_grid:
        if c_value <= 0:
            raise ValueError("Every regularization-grid C value must be positive.")
        model = LogisticRegression(
            C=float(c_value),
            solver="liblinear",
            max_iter=2_000,
            random_state=random_seed,
        )
        model.fit(train.matrix, train.targets)
        probability = model.predict_proba(validation.matrix)[:, 1]
        auroc = float(roc_auc_score(validation.targets, probability))
        records.append({"C": float(c_value), "validation_auroc": auroc})
        models[float(c_value)] = model
    best = min(records, key=lambda row: (-row["validation_auroc"], row["C"]))
    return models[best["C"]], records, best["C"]


def run_count_baseline(
    event_path: Path,
    cohort_path: Path,
    *,
    tasks: Sequence[str],
    private_root: Path = Path("outputs/private/count_baseline"),
    report_root: Path = Path("reports/generated/count_baseline"),
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    random_seed: int = 20260815,
    minimum_report_group_size: int = 20,
) -> dict[str, Any]:
    """Train and evaluate one validation-only count model per task."""
    validate_requested_splits(("train", "validation"))
    validate_cohort_splits(cohort_path)
    private_root.mkdir(parents=True, exist_ok=True)
    task_reports: dict[str, Any] = {}
    for task in tasks:
        task_root = private_root / task
        task_root.mkdir(parents=True, exist_ok=True)
        vocabulary = fit_vocabulary(event_path, cohort_path, task=task)
        train = build_sparse_counts(event_path, cohort_path, task=task, split="train", vocabulary=vocabulary)
        validation = build_sparse_counts(
            event_path,
            cohort_path,
            task=task,
            split="validation",
            vocabulary=vocabulary,
        )
        model, candidates, selected_c = _fit_candidates(
            train,
            validation,
            c_grid=c_grid,
            random_seed=random_seed,
        )
        validation_probability = model.predict_proba(validation.matrix)[:, 1]
        train_prevalence = float(train.targets.mean())
        metrics = validation_metrics(
            validation.targets,
            validation_probability,
            reference_probability=train_prevalence,
            minimum_group_size=minimum_report_group_size,
        )

        sparse.save_npz(task_root / "train_counts_log1p.npz", train.matrix, compressed=True)
        sparse.save_npz(task_root / "validation_counts_log1p.npz", validation.matrix, compressed=True)
        np.save(task_root / "train_targets.npy", train.targets, allow_pickle=False)
        np.save(task_root / "validation_targets.npy", validation.targets, allow_pickle=False)
        np.save(task_root / "validation_probabilities.npy", validation_probability, allow_pickle=False)
        pq.write_table(
            pa.table({"instance_key": train.instance_keys, "target": train.targets.astype(bool)}),
            task_root / "train_instances.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.table(
                {
                    "instance_key": validation.instance_keys,
                    "target": validation.targets.astype(bool),
                    "probability": validation_probability,
                }
            ),
            task_root / "validation_predictions.parquet",
            compression="zstd",
        )
        (task_root / "vocabulary.json").write_text(json.dumps(vocabulary, sort_keys=True) + "\n", encoding="utf-8")
        joblib.dump(model, task_root / "model.joblib")

        task_reports[task] = {
            "estimate_split": "validation",
            "train_instances": int(train.matrix.shape[0]),
            "validation_instances": int(validation.matrix.shape[0]),
            "vocabulary_size": len(vocabulary),
            "transform": "log1p applied to nonzero per-instance code counts",
            "candidate_results": candidates,
            "selected_C": selected_c,
            "tie_rule": "highest validation AUROC; smallest C wins an exact tie",
            "metrics": metrics,
        }

    report: dict[str, Any] = {
        "name": "simple count-based reference",
        "status": "PASS",
        "allowed_modeling_splits": ["train", "validation"],
        "test_split_inspected": False,
        "feature_rule": "structured-code counts from events satisfying event_time < prediction_time",
        "vocabulary_fit_split": "train",
        "regularization_grid": [float(value) for value in c_grid],
        "tasks": task_reports,
    }
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(report_text)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "validation_results.json").write_text(report_text, encoding="utf-8")
    return report
