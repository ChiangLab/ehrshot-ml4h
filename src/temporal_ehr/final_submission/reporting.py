"""Frozen readouts, shared patient-cluster bootstrap, and held-out aggregate report."""

from __future__ import annotations

import json
import math
import os
import resource
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.features.count_baseline import build_sparse_counts
from temporal_ehr.final_m1.embedding import condition_label, load_final_condition_embeddings
from temporal_ehr.final_m1.foundation import _write_immutable_json
from temporal_ehr.final_m1.report import _primary_scopes
from temporal_ehr.final_submission.protocol import load_frozen_manifest
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.pilot.evaluation import calibration_summary, metric_set
from temporal_ehr.pilot.model import load_model_identity
from temporal_ehr.serialization.integrity import stable_hash

ConditionData = tuple[list[dict[str, Any]], np.ndarray]


class BootstrapData(NamedTuple):
    positions: np.ndarray
    targets: np.ndarray
    probabilities: np.ndarray


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists() or not rows:
        raise ContractError("Refusing to overwrite or write empty held-out predictions.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".parquet", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), temporary, compression="zstd")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mapping(data: ConditionData, split: str) -> dict[str, tuple[dict[str, Any], np.ndarray]]:
    metadata, vectors = data
    return {
        str(row["instance_key"]): (row, vectors[int(row["condition_embedding_row_index"])])
        for row in metadata
        if row["split"] == split
    }


def _fit_fixed_scope(
    *,
    task: str,
    scope: str,
    train_data: Mapping[str, ConditionData],
    test_data: Mapping[str, ConditionData],
    selected_c: Mapping[str, float],
    random_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = tuple(train_data)
    if not labels or set(labels) != set(test_data) or set(labels) != set(selected_c):
        raise ContractError(f"Frozen scope {task}/{scope} has incomplete condition or hyperparameter coverage.")
    train_maps = {label: _mapping(train_data[label], "train") for label in labels}
    test_maps = {label: _mapping(test_data[label], "test") for label in labels}
    common_train = set.intersection(*(set(value) for value in train_maps.values()))
    common_test = set.intersection(*(set(value) for value in test_maps.values()))
    if not common_train or not common_test:
        raise ContractError(f"Frozen scope {task}/{scope} has no aligned train or test patients.")
    report: dict[str, Any] = {
        "fit_split": "train",
        "evaluation_split": "test",
        "train_instances": len(common_train),
        "test_instances": len(common_test),
        "representations": {},
    }
    predictions: list[dict[str, Any]] = []
    reference: dict[str, tuple[bool, ...]] = {}
    for split, maps, keys in (
        ("train", train_maps, sorted(common_train)),
        ("test", test_maps, sorted(common_test)),
    ):
        outcomes = {
            label: tuple(bool(maps[label][key][0]["target"]) for key in keys)
            for label in labels
        }
        if len(set(outcomes.values())) != 1:
            raise ContractError(f"Frozen scope {task}/{scope} has condition label drift on {split}.")
        reference[split] = next(iter(outcomes.values()))
    for label in labels:
        train_keys = sorted(common_train)
        test_keys = sorted(common_test)
        train_matrix = np.asarray([train_maps[label][key][1] for key in train_keys], dtype=np.float32)
        test_matrix = np.asarray([test_maps[label][key][1] for key in test_keys], dtype=np.float32)
        train_targets = np.asarray(reference["train"], dtype=np.int8)
        test_targets = np.asarray(reference["test"], dtype=np.int8)
        c_value = float(selected_c[label])
        model = LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=2000,
            random_state=random_seed,
        )
        model.fit(train_matrix, train_targets)
        probabilities = model.predict_proba(test_matrix)[:, 1]
        report["representations"][label] = {
            "selected_C": c_value,
            "selection_source": "hash-pinned validation report",
            "refit_on_train_plus_validation": False,
            "metrics": metric_set(test_targets, probabilities),
            "calibration": calibration_summary(test_targets, probabilities),
        }
        representation = label.split(":seed=", 1)[0]
        seed = int(label.rsplit("=", 1)[-1]) if ":seed=" in label else None
        for key, target, probability in zip(test_keys, test_targets, probabilities, strict=True):
            metadata = test_maps[label][key][0]
            predictions.append(
                {
                    "task": task,
                    "scope": scope,
                    "representation": representation,
                    "perturbation_seed": seed,
                    "instance_key": key,
                    "subject_id": int(metadata["subject_id"]),
                    "target": bool(target),
                    "probability": float(probability),
                }
            )
    return report, predictions


def _prediction_map(
    predictions: Sequence[Mapping[str, Any]],
    *,
    task: str,
    scope: str,
    representation: str,
    seed: int | None = None,
) -> dict[int, tuple[bool, float]]:
    result = {
        int(row["subject_id"]): (bool(row["target"]), float(row["probability"]))
        for row in predictions
        if row["task"] == task
        and row["scope"] == scope
        and row["representation"] == representation
        and row.get("perturbation_seed") == seed
    }
    if not result:
        raise ContractError(f"Missing private predictions for {task}/{scope}/{representation}/{seed}.")
    return result


def _add_c3_aggregates(
    predictions: list[dict[str, Any]],
    tasks: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for task in tasks:
        correct_maps = [
            _prediction_map(
                predictions,
                task=task,
                scope=f"c3_seed_{seed}",
                representation="C1_correct_compact_time",
            )
            for seed in seeds
        ]
        shuffled_maps = [
            _prediction_map(
                predictions,
                task=task,
                scope=f"c3_seed_{seed}",
                representation="C3_shuffled_earlier_time_labels",
                seed=seed,
            )
            for seed in seeds
        ]
        common = set.intersection(*(set(value) for value in [*correct_maps, *shuffled_maps]))
        if not common:
            raise ContractError(f"No common held-out C3 patient set for {task}.")
        seed_aurocs = []
        for seed, shuffled in zip(seeds, shuffled_maps, strict=True):
            keys = sorted(common)
            targets = np.asarray([shuffled[key][0] for key in keys], dtype=np.int8)
            seed_aurocs.append(
                {
                    "seed": int(seed),
                    "auroc": float(roc_auc_score(targets, [shuffled[key][1] for key in keys])),
                }
            )
        for subject in sorted(common):
            targets = {mapping[subject][0] for mapping in [*correct_maps, *shuffled_maps]}
            if len(targets) != 1:
                raise ContractError("C3 seed aggregation encountered target drift.")
            target = targets.pop()
            predictions.extend(
                [
                    {
                        "task": task,
                        "scope": "c3_three_seed_aggregate",
                        "representation": "C1_correct_compact_time",
                        "perturbation_seed": None,
                        "instance_key": f"aggregate:{task}:{subject}",
                        "subject_id": subject,
                        "target": target,
                        "probability": float(np.mean([mapping[subject][1] for mapping in correct_maps])),
                    },
                    {
                        "task": task,
                        "scope": "c3_three_seed_aggregate",
                        "representation": "C3_three_seed_probability_mean",
                        "perturbation_seed": None,
                        "instance_key": f"aggregate:{task}:{subject}",
                        "subject_id": subject,
                        "target": target,
                        "probability": float(np.mean([mapping[subject][1] for mapping in shuffled_maps])),
                    },
                ]
            )
        values = np.asarray([item["auroc"] for item in seed_aurocs], dtype=np.float64)
        reports[task] = {
            "seeds": seed_aurocs,
            "mean_seed_auroc": float(values.mean()),
            "seed_auroc_standard_deviation": float(values.std(ddof=1)),
            "aggregation": "arithmetic mean of three frozen-seed probabilities before inference",
            "instances": len(common),
        }
    return reports


def _evaluate_count_and_prevalence(
    *,
    event_path: Path,
    task: str,
    spec: Any,
    selected_c: float,
    c1_test: ConditionData,
    primary_identity: Any,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    validation_root = spec.validation.private_root / "count_baseline" / "tasks" / task
    vocabulary_path = validation_root / "vocabulary.json"
    model_path = validation_root / "model.joblib"
    if not vocabulary_path.is_file() or not model_path.is_file():
        raise ContractError(f"Frozen B3 training artifacts are missing for {task}.")
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    if not math.isclose(float(model.C), selected_c, rel_tol=0.0, abs_tol=1e-12):
        raise ContractError(f"Frozen B3 model C drifted for {task}.")
    counts = build_sparse_counts(
        event_path,
        spec.test_cohort_path,
        task=task,
        split="test",
        vocabulary=vocabulary,
        allow_test=True,
    )
    probabilities = model.predict_proba(counts.matrix)[:, 1]
    metadata, _ = c1_test
    test_rows = {str(row["instance_key"]): row for row in metadata if row["split"] == "test"}
    train_metadata, _ = load_final_condition_embeddings(
        spec=spec.validation,
        task=task,
        label="C1_correct_compact_time",
        role="primary",
        identity=primary_identity,
    )
    train_prevalence = float(np.mean([bool(row["target"]) for row in train_metadata if row["split"] == "train"]))
    if set(counts.instance_keys) != set(test_rows):
        raise ContractError(f"Held-out B3 rows do not align with C1 for {task}.")
    predictions: list[dict[str, Any]] = []
    for key, target, probability in zip(counts.instance_keys, counts.targets, probabilities, strict=True):
        row = test_rows[key]
        if bool(row["target"]) != bool(target):
            raise ContractError("Held-out B3 target drifted from M1 metadata.")
        common = {
            "task": task,
            "scope": "complete_cohort",
            "instance_key": key,
            "subject_id": int(row["subject_id"]),
            "target": bool(target),
            "perturbation_seed": None,
        }
        predictions.extend(
            [
                {**common, "representation": "B3_code_count", "probability": float(probability)},
                {
                    **common,
                    "representation": "B4_training_prevalence",
                    "probability": train_prevalence,
                },
            ]
        )
    b3 = {
        "selected_C": selected_c,
        "selection_source": "hash-pinned validation report",
        "vocabulary_fit_split": "train",
        "vocabulary_size": len(vocabulary),
        "test_instances": len(counts.targets),
        "metrics": metric_set(counts.targets, probabilities),
        "calibration": calibration_summary(counts.targets, probabilities),
    }
    prevalence_probabilities = np.full(len(counts.targets), train_prevalence)
    b4 = {
        "training_prevalence": train_prevalence,
        "test_instances": len(counts.targets),
        "metrics": metric_set(counts.targets, prevalence_probabilities),
        "calibration": calibration_summary(counts.targets, prevalence_probabilities),
    }
    return b3, b4, predictions


def _holm(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    result: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * values[key]))
        result[key] = running
    return result


def _benjamini_hochberg(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, key in reversed(list(enumerate(ordered, start=1))):
        running = min(running, values[key] * total / rank)
        adjusted[key] = min(1.0, running)
    return adjusted


def _two_sided_probability(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    lower = (np.count_nonzero(array <= 0.0) + 1) / (len(array) + 1)
    upper = (np.count_nonzero(array >= 0.0) + 1) / (len(array) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def _bootstrap_data(
    mapping: Mapping[int, tuple[bool, float]],
    patient_positions: Mapping[int, int],
) -> BootstrapData:
    patients = sorted(mapping)
    return BootstrapData(
        positions=np.asarray([patient_positions[patient] for patient in patients], dtype=np.int64),
        targets=np.asarray([mapping[patient][0] for patient in patients], dtype=np.int8),
        probabilities=np.asarray([mapping[patient][1] for patient in patients], dtype=np.float64),
    )


def _resampled_metrics(data: BootstrapData, counts: np.ndarray) -> dict[str, float] | None:
    weights = counts[data.positions]
    indices = np.repeat(np.arange(len(weights)), weights)
    targets = data.targets[indices]
    if len(np.unique(targets)) != 2:
        return None
    return metric_set(targets, data.probabilities[indices])


def _shared_cluster_bootstrap(
    predictions: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    replicates: int,
    seed: int,
    secondary_names: Sequence[str],
) -> dict[str, Any]:
    patients = sorted({int(row["subject_id"]) for row in predictions})
    patient_positions = {patient: index for index, patient in enumerate(patients)}
    main = {
        task: _bootstrap_data(
            _prediction_map(
                predictions,
                task=task,
                scope="complete_cohort",
                representation="C1_correct_compact_time",
            ),
            patient_positions,
        )
        for task in tasks
    }
    contrast_specs = {
        "C1_minus_B1": (
            "b1_aligned",
            "C1_correct_compact_time",
            "B1_latest_window_only",
        ),
        "C1_minus_C3_mean": (
            "c3_three_seed_aggregate",
            "C1_correct_compact_time",
            "C3_three_seed_probability_mean",
        ),
        "C1_minus_C0": ("complete_cohort", "C1_correct_compact_time", "C0_order_only"),
        "C1_minus_C2": (
            "c2_perturbable",
            "C1_correct_compact_time",
            "C2_reversed_earlier_gaps",
        ),
        "C1_minus_B2": (
            "b2_aligned",
            "C1_correct_compact_time",
            "B2_latest_code_occurrence",
        ),
        "C1_minus_B3": ("complete_cohort", "C1_correct_compact_time", "B3_code_count"),
        "C1_minus_B4": (
            "complete_cohort",
            "C1_correct_compact_time",
            "B4_training_prevalence",
        ),
    }
    contrast_data: dict[tuple[str, str], tuple[BootstrapData, BootstrapData]] = {}
    for name, (scope, first, second) in contrast_specs.items():
        for task in tasks:
            first_map = _prediction_map(predictions, task=task, scope=scope, representation=first)
            second_map = _prediction_map(predictions, task=task, scope=scope, representation=second)
            if set(first_map) != set(second_map) or any(first_map[key][0] != second_map[key][0] for key in first_map):
                raise ContractError(f"Held-out paired contrast alignment failed for {task}/{name}.")
            contrast_data[(task, name)] = (
                _bootstrap_data(first_map, patient_positions),
                _bootstrap_data(second_map, patient_positions),
            )

    main_distributions = {
        task: {metric: [] for metric in ("auroc", "auprc", "brier_score", "log_loss")}
        for task in tasks
    }
    macro_distributions = {metric: [] for metric in ("auroc", "auprc", "brier_score", "log_loss")}
    contrast_distributions = {
        (task, name): [] for task in tasks for name in contrast_specs
    }
    panel_contrast_distributions = {name: [] for name in contrast_specs}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        drawn = rng.integers(0, len(patients), size=len(patients))
        counts = np.bincount(drawn, minlength=len(patients))
        task_metrics: dict[str, dict[str, float]] = {}
        for task in tasks:
            metrics = _resampled_metrics(main[task], counts)
            if metrics is not None:
                task_metrics[task] = metrics
                for metric, value in metrics.items():
                    main_distributions[task][metric].append(value)
        if len(task_metrics) == len(tasks):
            for metric in macro_distributions:
                macro_distributions[metric].append(float(np.mean([task_metrics[task][metric] for task in tasks])))
        for name in contrast_specs:
            task_values = []
            for task in tasks:
                first, second = contrast_data[(task, name)]
                first_metrics = _resampled_metrics(first, counts)
                second_metrics = _resampled_metrics(second, counts)
                if first_metrics is None or second_metrics is None:
                    continue
                value = float(first_metrics["auroc"] - second_metrics["auroc"])
                contrast_distributions[(task, name)].append(value)
                task_values.append(value)
            if len(task_values) == len(tasks):
                panel_contrast_distributions[name].append(float(np.mean(task_values)))

    minimum_valid = max(20, replicates // 2)

    def interval(values: Sequence[float]) -> list[float]:
        if len(values) < minimum_valid:
            raise ContractError("Too few valid shared patient-cluster bootstrap replicates.")
        return [float(value) for value in np.percentile(np.asarray(values), [2.5, 97.5])]

    main_intervals = {
        task: {
            metric: {"ci_95": interval(values), "valid_replicates": len(values)}
            for metric, values in distributions.items()
        }
        for task, distributions in main_distributions.items()
    }
    macro_intervals = {
        metric: {"ci_95": interval(values), "valid_replicates": len(values)}
        for metric, values in macro_distributions.items()
    }
    task_contrasts: dict[str, dict[str, Any]] = {task: {} for task in tasks}
    secondary_p: dict[str, float] = {}
    for task in tasks:
        for name in contrast_specs:
            first, second = contrast_data[(task, name)]
            observed = float(
                roc_auc_score(first.targets, first.probabilities)
                - roc_auc_score(second.targets, second.probabilities)
            )
            values = contrast_distributions[(task, name)]
            key = f"{task}:{name}"
            task_contrasts[task][name] = {
                "estimate": observed,
                "ci_95": interval(values),
                "unadjusted_two_sided_p": _two_sided_probability(values),
                "valid_replicates": len(values),
                "instances": len(first.targets),
            }
            if name in secondary_names:
                secondary_p[key] = task_contrasts[task][name]["unadjusted_two_sided_p"]
    adjusted_secondary = _benjamini_hochberg(secondary_p)
    for key, adjusted in adjusted_secondary.items():
        task, name = key.split(":", 1)
        task_contrasts[task][name]["benjamini_hochberg_q"] = adjusted

    panel_contrasts = {}
    for name, values in panel_contrast_distributions.items():
        observed = float(np.mean([task_contrasts[task][name]["estimate"] for task in tasks]))
        panel_contrasts[name] = {
            "estimate": observed,
            "ci_95": interval(values),
            "unadjusted_two_sided_p": _two_sided_probability(values),
            "valid_replicates": len(values),
            "task_weighting": "equal",
        }
    co_primary = {name: panel_contrasts[name]["unadjusted_two_sided_p"] for name in ("C1_minus_B1", "C1_minus_C3_mean")}
    for name, adjusted in _holm(co_primary).items():
        panel_contrasts[name]["holm_adjusted_p"] = adjusted
    return {
        "requested_replicates": replicates,
        "seed": seed,
        "resampling_unit": "held-out patient cluster across tasks",
        "global_patient_clusters": len(patients),
        "identical_resamples_across_estimands": True,
        "main_C1_metric_intervals_by_task": main_intervals,
        "main_C1_panel_macro_intervals": macro_intervals,
        "task_contrasts": task_contrasts,
        "equal_weight_panel_macro_contrasts": panel_contrasts,
        "co_primary_family": ["C1_minus_B1", "C1_minus_C3_mean"],
        "co_primary_correction": "Holm",
        "per_task_secondary_family": list(secondary_names),
        "per_task_secondary_correction": "Benjamini-Hochberg",
    }


def _macro(task_reports: Mapping[str, Any], tasks: Sequence[str]) -> dict[str, Any]:
    locators = {
        "C0_complete_cohort": ("scope", "complete_cohort", "C0_order_only"),
        "C1_complete_cohort": ("scope", "complete_cohort", "C1_correct_compact_time"),
        "B1_latest_window": ("scope", "b1_aligned", "B1_latest_window_only"),
        "B2_latest_code": ("scope", "b2_aligned", "B2_latest_code_occurrence"),
        "C2_perturbable": ("scope", "c2_perturbable", "C2_reversed_earlier_gaps"),
        "C3_three_seed_probability_mean": ("aggregate", "c3_aggregate", "metrics"),
        "B3_code_count": ("baseline", "B3_code_count", "metrics"),
        "B4_training_prevalence": ("baseline", "B4_training_prevalence", "metrics"),
    }
    result = {}
    for name, (kind, location, label) in locators.items():
        values = []
        for task in tasks:
            report = task_reports[task]
            if kind == "scope":
                metrics = report["primary_scopes"][location]["representations"][label]["metrics"]
            else:
                metrics = report[location][label]
            values.append(metrics)
        result[name] = {
            "task_weighting": "equal",
            "task_count": len(tasks),
            "metrics": {
                metric: float(np.mean([value[metric] for value in values]))
                for metric in ("auroc", "auprc", "brier_score", "log_loss")
            },
        }
    return result


def fit_m1_held_out_and_report(
    *,
    repository_root: Path,
    config_path: Path,
    event_path: Path,
    model_root: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Run the frozen train-fit/test-evaluate readouts once and write one aggregate."""
    started = time.perf_counter()
    root = repository_root.resolve()
    spec, manifest = load_frozen_manifest(root, config_path, source_git_commit)
    destination = spec.aggregate_root / "m1_held_out_results.json"
    predictions_path = spec.private_root / "fit" / "test_predictions.parquet"
    checkpoint_path = spec.private_root / "fit" / "AGGREGATE_CHECKPOINT.json"
    if destination.exists():
        raise ContractError("One-time held-out aggregate already exists; refusing rerun.")
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("status") != "PASS"
            or checkpoint.get("source_git_commit") != source_git_commit
            or checkpoint.get("frozen_manifest_sha256") != manifest["frozen_manifest_sha256"]
            or not predictions_path.is_file()
            or checkpoint.get("private_predictions_sha256") != file_sha256(predictions_path)
            or checkpoint.get("private_prediction_rows") != pq.read_metadata(predictions_path).num_rows
        ):
            raise ContractError("Private held-out fit checkpoint failed identity, hash, or row-count validation.")
        assert_privacy_safe_report(json.dumps(checkpoint, sort_keys=True))
        _write_immutable_json(destination, checkpoint, aggregate=True)
        return checkpoint
    selected = manifest["selected_hyperparameters"]
    identity = load_model_identity(
        model_root / spec.validation.primary_model_directory_name,
        expected_repository=spec.validation.primary_model_repository,
    )
    scopes = _primary_scopes(spec.validation)
    all_predictions: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}
    for task in spec.tasks:
        test_texts = pq.read_table(spec.private_root / "prepared" / task / "texts.parquet").to_pylist()
        available = sorted({condition_label(row) for row in test_texts})
        train_data = {
            label: load_final_condition_embeddings(
                spec=spec.validation,
                task=task,
                label=label,
                role="primary",
                identity=identity,
            )
            for label in available
        }
        test_data = {
            label: load_final_condition_embeddings(
                spec=spec.validation,
                task=task,
                label=label,
                role="primary",
                identity=identity,
                artifact_experiment_id=spec.experiment_id,
                cache_base=spec.private_root / "embeddings",
            )
            for label in available
        }
        scope_reports: dict[str, Any] = {}
        task_predictions: list[dict[str, Any]] = []
        for scope, labels in scopes.items():
            report, rows = _fit_fixed_scope(
                task=task,
                scope=scope,
                train_data={label: train_data[label] for label in labels},
                test_data={label: test_data[label] for label in labels},
                selected_c=selected[task]["scopes"][scope],
                random_seed=int(spec.validation.raw["logistic_seed"]),
            )
            scope_reports[scope] = report
            task_predictions.extend(rows)
        b3, b4, baseline_predictions = _evaluate_count_and_prevalence(
            event_path=event_path,
            task=task,
            spec=spec,
            selected_c=float(selected[task]["B3_code_count"]),
            c1_test=test_data["C1_correct_compact_time"],
            primary_identity=identity,
        )
        task_predictions.extend(baseline_predictions)
        task_reports[task] = {
            "primary_scopes": scope_reports,
            "B3_code_count": b3,
            "B4_training_prevalence": b4,
        }
        all_predictions.extend(task_predictions)
    c3_variability = _add_c3_aggregates(all_predictions, spec.tasks, spec.validation.c3_seeds)
    for task in spec.tasks:
        c3_rows = _prediction_map(
            all_predictions,
            task=task,
            scope="c3_three_seed_aggregate",
            representation="C3_three_seed_probability_mean",
        )
        keys = sorted(c3_rows)
        targets = np.asarray([c3_rows[key][0] for key in keys], dtype=np.int8)
        probabilities = np.asarray([c3_rows[key][1] for key in keys], dtype=np.float64)
        task_reports[task]["c3_aggregate"] = {
            "metrics": metric_set(targets, probabilities),
            "calibration": calibration_summary(targets, probabilities),
            "seed_variability": c3_variability[task],
        }
    prediction_keys = [
        (
            row["task"],
            row["scope"],
            row["representation"],
            row.get("perturbation_seed"),
            row["instance_key"],
        )
        for row in all_predictions
    ]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise ContractError("Held-out private predictions are not unique by task/scope/condition/instance.")
    if predictions_path.is_file():
        existing_predictions = pq.read_table(predictions_path).to_pylist()
        if stable_hash(existing_predictions) != stable_hash(all_predictions):
            raise ContractError("Partial held-out predictions differ from deterministic same-manifest recovery.")
    else:
        _atomic_parquet(predictions_path, all_predictions)
    secondary_names = ("C1_minus_C0", "C1_minus_C2", "C1_minus_B2", "C1_minus_B3", "C1_minus_B4")
    bootstrap = _shared_cluster_bootstrap(
        all_predictions,
        tasks=spec.tasks,
        replicates=spec.bootstrap_replicates,
        seed=spec.bootstrap_seed,
        secondary_names=secondary_names,
    )
    cohort_complete = json.loads((spec.test_cohort_path.parent / "COMPLETE.json").read_text(encoding="utf-8"))
    preparation = [
        json.loads((spec.private_root / "prepared" / task / "COMPLETE.json").read_text(encoding="utf-8"))[
            "private_summary"
        ]
        for task in spec.tasks
    ]
    m3 = json.loads(spec.m3_report_path.read_text(encoding="utf-8"))
    report = {
        "status": "PASS",
        "protocol_decision": "M1_PRIMARY_ONE_TIME_HELD_OUT_COMPLETE_NO_DESIGN_CHANGES_PERMITTED",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "frozen_manifest_sha256": manifest["frozen_manifest_sha256"],
        "scope": spec.raw["approach_scope"],
        "tasks": task_reports,
        "equal_weight_panel_macro": _macro(task_reports, spec.tasks),
        "shared_patient_cluster_bootstrap": bootstrap,
        "held_out_cohort": cohort_complete["private_summary"],
        "held_out_preparation_diagnostics": preparation,
        "primary_model": {
            "repository": identity.repository,
            "artifact_hash": identity.artifact_hash,
            "tokenizer_hash": identity.tokenizer_hash,
            "token_budget": spec.validation.token_budget,
            "frozen": True,
            "full_dimension": True,
            "l2_normalized_float32": True,
        },
        "fitting": spec.raw["fitting"],
        "m3_secondary_validation_only": {
            "status": m3["status"],
            "protocol_decision": m3["protocol_decision"],
            "report_sha256": spec.m3_report_sha256,
            "equal_weight_panel_macro": m3.get("equal_weight_panel_macro"),
            "training_seeds": m3.get("training_seeds"),
            "task_seed_aggregation_and_variability": {
                task: {
                    "averaged_seed_metrics": m3["tasks"][task]["conditions"][
                        "C1_correct_compact_time"
                    ]["averaged_seed_metrics"],
                    "seed_auroc_standard_deviation": m3["tasks"][task]["conditions"][
                        "C1_correct_compact_time"
                    ]["seed_auroc_standard_deviation"],
                    "three_seed_probability_average": m3["tasks"][task]["conditions"][
                        "C1_correct_compact_time"
                    ]["three_seed_probability_average"],
                }
                for task in spec.tasks
            },
            "held_out_accessed": False,
        },
        "m2": {
            "status": "INCOMPLETE_FUTURE_WORK",
            "reason": "Repeated bounded-summary runtime and fidelity failures before the deadline",
            "held_out_accessed": False,
        },
        "validation_evidence": spec.raw["validation_evidence"],
        "private_predictions_sha256": file_sha256(predictions_path),
        "private_prediction_rows": len(all_predictions),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "test_evaluated_once": True,
        "protocol_modified_after_test_access": False,
        "test_performance_written_to_job_log": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(checkpoint_path, report)
    _write_immutable_json(destination, report, aggregate=True)
    return report
