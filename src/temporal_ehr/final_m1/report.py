"""Train/validation-only final M1 readouts, contrasts, macro estimates, and report."""

from __future__ import annotations

import json
import os
import resource
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import FinalM1Authorization, load_final_m1_authorization
from temporal_ehr.final_m1.embedding import condition_label, load_final_condition_embeddings
from temporal_ehr.final_m1.foundation import _write_immutable_json
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.pilot.evaluation import (
    calibration_summary,
    fit_logistic_readout,
    metric_set,
    paired_bootstrap_interval,
)
from temporal_ehr.pilot.model import ModelIdentity, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash

ConditionData = tuple[list[dict[str, Any]], np.ndarray]


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ContractError("Refusing to write an empty final prediction table.")
    if path.exists():
        raise ContractError("Refusing to overwrite final validation predictions.")
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


def _condition_mapping(data: ConditionData, split: str) -> dict[str, tuple[dict[str, Any], np.ndarray]]:
    metadata, vectors = data
    return {
        str(row["instance_key"]): (row, vectors[int(row["condition_embedding_row_index"])])
        for row in metadata
        if row["split"] == split
    }


def fit_condition_scope(
    *,
    task: str,
    scope: str,
    condition_data: Mapping[str, ConditionData],
    c_grid: Sequence[float],
    random_seed: int,
    model_role: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit every condition on identical patient sets within one prespecified scope."""
    if not condition_data:
        raise ContractError(f"Final fitting scope {scope} contains no conditions.")
    mappings = {
        split: {label: _condition_mapping(data, split) for label, data in condition_data.items()}
        for split in ("train", "validation")
    }
    common = {
        split: set.intersection(*(set(rows) for rows in mappings[split].values()))
        for split in ("train", "validation")
    }
    if not common["train"] or not common["validation"]:
        raise ContractError(f"Final fitting scope {scope} has no aligned train/validation patients.")
    report: dict[str, Any] = {
        "train_instances": len(common["train"]),
        "validation_instances": len(common["validation"]),
        "resampling_unit": "patient (at most one selected instance per task)",
        "representations": {},
    }
    predictions: list[dict[str, Any]] = []
    reference_targets: dict[str, tuple[bool, ...]] = {}
    for split in ("train", "validation"):
        keys = sorted(common[split])
        targets_by_label = {
            label: tuple(bool(mappings[split][label][key][0]["target"]) for key in keys)
            for label in condition_data
        }
        if len(set(targets_by_label.values())) != 1:
            raise ContractError("Final condition patient/label alignment failed.")
        reference_targets[split] = next(iter(targets_by_label.values()))
    report["train_prevalence"] = float(np.mean(reference_targets["train"]))
    report["validation_prevalence"] = float(np.mean(reference_targets["validation"]))
    for label in condition_data:
        matrices = {}
        targets = {}
        keys_by_split = {}
        for split in ("train", "validation"):
            keys = sorted(common[split])
            keys_by_split[split] = keys
            matrices[split] = np.asarray([mappings[split][label][key][1] for key in keys], dtype=np.float32)
            targets[split] = np.asarray(reference_targets[split], dtype=np.int8)
        probability, fit_report = fit_logistic_readout(
            matrices["train"],
            targets["train"],
            matrices["validation"],
            targets["validation"],
            c_grid=c_grid,
            random_seed=random_seed,
        )
        report["representations"][label] = fit_report
        representation = label.split(":seed=", 1)[0]
        perturbation_seed = int(label.rsplit("=", 1)[-1]) if ":seed=" in label else None
        for key, target, value in zip(
            keys_by_split["validation"], targets["validation"], probability, strict=True
        ):
            predictions.append(
                {
                    "model_role": model_role,
                    "task": task,
                    "scope": scope,
                    "representation": representation,
                    "perturbation_seed": perturbation_seed,
                    "instance_key": key,
                    "target": bool(target),
                    "probability": float(value),
                }
            )
    return report, predictions


def _prediction_map(
    predictions: Sequence[Mapping[str, Any]],
    *,
    task: str,
    scope: str,
    model_role: str,
    label: str,
) -> dict[str, Mapping[str, Any]]:
    representation = label.split(":seed=", 1)[0]
    seed = int(label.rsplit("=", 1)[-1]) if ":seed=" in label else None
    return {
        str(row["instance_key"]): row
        for row in predictions
        if row["task"] == task
        and row["scope"] == scope
        and row["model_role"] == model_role
        and row["representation"] == representation
        and row.get("perturbation_seed") == seed
    }


def paired_contrast(
    predictions: Sequence[Mapping[str, Any]],
    *,
    task: str,
    scope: str,
    model_role: str,
    first: str,
    second: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    first_rows = _prediction_map(
        predictions,
        task=task,
        scope=scope,
        model_role=model_role,
        label=first,
    )
    second_rows = _prediction_map(
        predictions,
        task=task,
        scope=scope,
        model_role=model_role,
        label=second,
    )
    if not first_rows or set(first_rows) != set(second_rows):
        raise ContractError("Final paired contrast patient sets differ or are empty.")
    keys = sorted(first_rows)
    targets = np.asarray([first_rows[key]["target"] for key in keys], dtype=np.int8)
    if targets.tolist() != [second_rows[key]["target"] for key in keys]:
        raise ContractError("Final paired contrast labels differ.")
    return {
        "scope": scope,
        "first": first,
        "second": second,
        "instances": len(keys),
        **paired_bootstrap_interval(
            targets,
            np.asarray([first_rows[key]["probability"] for key in keys]),
            np.asarray([second_rows[key]["probability"] for key in keys]),
            replicates=replicates,
            seed=seed,
        ),
    }


def macro_paired_contrast(
    predictions: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    scope: str,
    first: str,
    second: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap patients within task and equally weight the four task differences."""
    task_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    observed: list[float] = []
    for task in tasks:
        first_rows = _prediction_map(
            predictions,
            task=task,
            scope=scope,
            model_role="primary",
            label=first,
        )
        second_rows = _prediction_map(
            predictions,
            task=task,
            scope=scope,
            model_role="primary",
            label=second,
        )
        if set(first_rows) != set(second_rows) or not first_rows:
            raise ContractError("Macro paired contrast lacks aligned patients for one task.")
        keys = sorted(first_rows)
        targets = np.asarray([first_rows[key]["target"] for key in keys], dtype=np.int8)
        first_probability = np.asarray([first_rows[key]["probability"] for key in keys])
        second_probability = np.asarray([second_rows[key]["probability"] for key in keys])
        task_arrays.append((targets, first_probability, second_probability))
        observed.append(float(roc_auc_score(targets, first_probability) - roc_auc_score(targets, second_probability)))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        task_estimates = []
        for targets, first_probability, second_probability in task_arrays:
            indices = rng.integers(0, len(targets), len(targets))
            sampled = targets[indices]
            if len(np.unique(sampled)) != 2:
                break
            task_estimates.append(
                float(
                    roc_auc_score(sampled, first_probability[indices])
                    - roc_auc_score(sampled, second_probability[indices])
                )
            )
        if len(task_estimates) == len(tasks):
            estimates.append(float(np.mean(task_estimates)))
    if len(estimates) < max(20, replicates // 2):
        raise ContractError("Too few valid equal-weight macro bootstrap replicates.")
    lower, upper = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return {
        "scope": scope,
        "first": first,
        "second": second,
        "metric": "equal-weight panel macro-AUROC difference",
        "estimate": float(np.mean(observed)),
        "ci_95": [float(lower), float(upper)],
        "requested_replicates": replicates,
        "valid_replicates": len(estimates),
        "resampling": "patients within task; four task estimates equally weighted",
    }


def _load_task_conditions(
    *,
    spec: FinalM1Authorization,
    task: str,
    role: str,
    identity: ModelIdentity,
    labels: Sequence[str],
) -> dict[str, ConditionData]:
    return {
        label: load_final_condition_embeddings(
            spec=spec,
            task=task,
            label=label,
            role=role,
            identity=identity,
        )
        for label in labels
    }


def _primary_scopes(spec: FinalM1Authorization) -> dict[str, tuple[str, ...]]:
    return {
        "complete_cohort": ("C0_order_only", "C1_correct_compact_time"),
        "b1_aligned": ("C1_correct_compact_time", "B1_latest_window_only"),
        "b2_aligned": ("C1_correct_compact_time", "B2_latest_code_occurrence"),
        "c2_perturbable": (
            "C0_order_only",
            "C1_correct_compact_time",
            "C2_reversed_earlier_gaps",
        ),
        **{
            f"c3_seed_{seed}": (
                "C0_order_only",
                "C1_correct_compact_time",
                f"C3_shuffled_earlier_time_labels:seed={seed}",
            )
            for seed in spec.c3_seeds
        },
    }


def _add_b3_b4(
    *,
    spec: FinalM1Authorization,
    task: str,
    task_predictions: list[dict[str, Any]],
    c1_data: ConditionData,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata, _ = c1_data
    train = {str(row["instance_key"]): row for row in metadata if row["split"] == "train"}
    validation = {str(row["instance_key"]): row for row in metadata if row["split"] == "validation"}
    train_prevalence = float(np.mean([bool(row["target"]) for row in train.values()]))
    validation_keys = sorted(validation)
    targets = np.asarray([bool(validation[key]["target"]) for key in validation_keys], dtype=np.int8)
    prevalence_probability = np.full(len(targets), train_prevalence)
    b4 = {
        "train_instances": len(train),
        "validation_instances": len(validation),
        "training_prevalence": train_prevalence,
        "metrics": metric_set(targets, prevalence_probability),
        "calibration": calibration_summary(targets, prevalence_probability),
    }
    count_report_path = spec.aggregate_root / "count_baseline" / "validation_results.json"
    count_predictions_path = spec.private_root / "count_baseline" / "tasks" / task / "validation_predictions.parquet"
    if not count_report_path.is_file() or not count_predictions_path.is_file():
        raise ContractError("Final B3 aggregate and private aligned predictions are required.")
    count_report = json.loads(count_report_path.read_text(encoding="utf-8"))
    if count_report.get("status") != "PASS" or count_report.get("test_split_inspected") is not False:
        raise ContractError("Final B3 is not a valid train/validation-only result.")
    count_rows = pq.read_table(count_predictions_path).to_pylist()
    count_map = {str(row["instance_key"]): row for row in count_rows}
    if len(count_map) != len(count_rows) or set(count_map) != set(validation):
        raise ContractError("Final B3 predictions do not align one-to-one with C1 validation patients.")
    for key in validation_keys:
        if bool(count_map[key]["target"]) != bool(validation[key]["target"]):
            raise ContractError("Final B3 labels drifted from final C1 metadata.")
    for key, target, prevalence in zip(validation_keys, targets, prevalence_probability, strict=True):
        task_predictions.extend(
            [
                {
                    "model_role": "primary",
                    "task": task,
                    "scope": "complete_cohort",
                    "representation": "B4_training_prevalence",
                    "perturbation_seed": None,
                    "instance_key": key,
                    "target": bool(target),
                    "probability": float(prevalence),
                },
                {
                    "model_role": "primary",
                    "task": task,
                    "scope": "complete_cohort",
                    "representation": "B3_code_count",
                    "perturbation_seed": None,
                    "instance_key": key,
                    "target": bool(target),
                    "probability": float(count_map[key]["probability"]),
                },
            ]
        )
    b3 = dict(count_report["tasks"][task])
    b3["calibration"] = calibration_summary(
        targets,
        np.asarray([float(count_map[key]["probability"]) for key in validation_keys]),
    )
    return b3, b4


def _macro_metric(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = ("auroc", "auprc", "brier_score", "log_loss")
    return {key: float(np.mean([float(value[key]) for value in values])) for key in keys}


def _panel_macro(task_reports: Mapping[str, Any], spec: FinalM1Authorization) -> dict[str, Any]:
    locators = {
        "C0_complete_cohort": ("scope", "complete_cohort", "C0_order_only"),
        "C1_complete_cohort": ("scope", "complete_cohort", "C1_correct_compact_time"),
        "B1_latest_window": ("scope", "b1_aligned", "B1_latest_window_only"),
        "B2_latest_code": ("scope", "b2_aligned", "B2_latest_code_occurrence"),
        "C2_perturbable": ("scope", "c2_perturbable", "C2_reversed_earlier_gaps"),
        "B3_code_count": ("baseline", "B3_code_count", ""),
        "B4_training_prevalence": ("baseline", "B4_training_prevalence", ""),
        **{
            f"C3_seed_{seed}_perturbable": (
                "scope",
                f"c3_seed_{seed}",
                f"C3_shuffled_earlier_time_labels:seed={seed}",
            )
            for seed in spec.c3_seeds
        },
    }
    output = {}
    for name, (kind, location, representation) in locators.items():
        metrics = []
        for task in spec.tasks:
            task_report = task_reports[task]
            if kind == "scope":
                metrics.append(task_report["primary_scopes"][location]["representations"][representation]["metrics"])
            elif location == "B3_code_count":
                metrics.append(task_report[location]["metrics"]["model"])
            else:
                metrics.append(task_report[location]["metrics"])
        output[name] = {
            "task_weighting": "equal",
            "task_count": len(spec.tasks),
            "metrics": _macro_metric(metrics),
        }
    return output


def fit_final_m1_and_report(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Fit final M1 readouts and write a privacy-safe aggregate with no test results."""
    started = time.perf_counter()
    spec = load_final_m1_authorization(config_path, repository_root=repository_root)
    foundation = json.loads((spec.aggregate_root / "foundation.json").read_text(encoding="utf-8"))
    if (
        foundation.get("status") != "PASS"
        or foundation.get("source_git_commit") != source_git_commit
        or foundation.get("content_matching_passed") is not True
        or foundation.get("test_predictions_or_performance_used") is not False
    ):
        raise ContractError("Final fit requires the same-source completed foundation gates.")
    throughput = json.loads((spec.aggregate_root / "throughput.json").read_text(encoding="utf-8"))
    if throughput.get("status") != "PASS" or throughput.get("source_git_commit") != source_git_commit:
        raise ContractError("Final fit requires the same-source completed throughput gate.")
    embedding_reports = {
        task: json.loads((spec.aggregate_root / "embedding" / f"{task}.json").read_text(encoding="utf-8"))
        for task in spec.tasks
    }
    if any(
        report.get("status") != "PASS"
        or report.get("source_git_commit") != source_git_commit
        or report.get("pilot_cache_reused") is not False
        or report.get("test_split_used") is not False
        for report in embedding_reports.values()
    ):
        raise ContractError("A final task embedding report failed identity, isolation, or split gates.")
    primary_identity = load_model_identity(
        model_root / spec.primary_model_directory_name,
        expected_repository=spec.primary_model_repository,
    )
    confirmatory_identity = load_model_identity(
        model_root / spec.confirmatory_model_directory_name,
        expected_repository=spec.confirmatory_model_repository,
    )
    c_grid = tuple(float(value) for value in spec.raw["regularization_grid"])
    logistic_seed = int(spec.raw["logistic_seed"])
    bootstrap_seed = int(spec.raw["bootstrap_seed"])
    replicates = int(spec.raw["bootstrap_replicates"])
    all_predictions: list[dict[str, Any]] = []
    task_reports: dict[str, Any] = {}
    scopes = _primary_scopes(spec)
    for task_index, task in enumerate(spec.tasks):
        prepared = read_final_task_texts(spec, task)
        available_labels = sorted({condition_label(row) for row in prepared})
        primary_data = _load_task_conditions(
            spec=spec,
            task=task,
            role="primary",
            identity=primary_identity,
            labels=available_labels,
        )
        scope_reports = {}
        task_predictions: list[dict[str, Any]] = []
        for scope, labels in scopes.items():
            selected = {label: primary_data[label] for label in labels}
            scope_report, predictions = fit_condition_scope(
                task=task,
                scope=scope,
                condition_data=selected,
                c_grid=c_grid,
                random_seed=logistic_seed,
                model_role="primary",
            )
            scope_reports[scope] = scope_report
            task_predictions.extend(predictions)
        b3, b4 = _add_b3_b4(
            spec=spec,
            task=task,
            task_predictions=task_predictions,
            c1_data=primary_data["C1_correct_compact_time"],
        )
        contrast_specs = [
            ("complete_cohort", "C1_correct_compact_time", "C0_order_only"),
            ("c2_perturbable", "C1_correct_compact_time", "C2_reversed_earlier_gaps"),
            ("b1_aligned", "C1_correct_compact_time", "B1_latest_window_only"),
            ("b2_aligned", "C1_correct_compact_time", "B2_latest_code_occurrence"),
            ("complete_cohort", "C1_correct_compact_time", "B3_code_count"),
            ("complete_cohort", "C1_correct_compact_time", "B4_training_prevalence"),
        ]
        contrasts = [
            paired_contrast(
                task_predictions,
                task=task,
                scope=scope,
                model_role="primary",
                first=first,
                second=second,
                replicates=replicates,
                seed=bootstrap_seed + task_index * 100 + index,
            )
            for index, (scope, first, second) in enumerate(contrast_specs)
        ]
        c3_contrasts = [
            paired_contrast(
                task_predictions,
                task=task,
                scope=f"c3_seed_{seed}",
                model_role="primary",
                first="C1_correct_compact_time",
                second=f"C3_shuffled_earlier_time_labels:seed={seed}",
                replicates=replicates,
                seed=bootstrap_seed + int(seed) + task_index,
            )
            for seed in spec.c3_seeds
        ]
        selected_priorities = tuple(
            throughput["feasibility_plan"]["tasks"][task]["confirmatory_priorities_selected"]
        )
        confirmatory_scopes: dict[str, Any] = {}
        confirmatory_predictions: list[dict[str, Any]] = []
        if selected_priorities:
            selected_labels = [
                label
                for label in available_labels
                if label.split(":seed=", 1)[0] in selected_priorities
            ]
            confirmatory_data = _load_task_conditions(
                spec=spec,
                task=task,
                role="confirmatory",
                identity=confirmatory_identity,
                labels=selected_labels,
            )
            # Standalone estimates for every runtime-authorized cell.
            for label in selected_labels:
                scope = f"standalone_{label.replace(':seed=', '_seed_')}"
                scope_report, predictions = fit_condition_scope(
                    task=task,
                    scope=scope,
                    condition_data={label: confirmatory_data[label]},
                    c_grid=c_grid,
                    random_seed=logistic_seed,
                    model_role="confirmatory",
                )
                confirmatory_scopes[scope] = scope_report
                confirmatory_predictions.extend(predictions)
            # Paired confirmatory scopes exist only when both priority cells were admitted.
            if "C1_correct_compact_time" in confirmatory_data:
                for label in selected_labels:
                    if label == "C1_correct_compact_time":
                        continue
                    scope = f"paired_C1_vs_{label.replace(':seed=', '_seed_')}"
                    scope_report, predictions = fit_condition_scope(
                        task=task,
                        scope=scope,
                        condition_data={
                            "C1_correct_compact_time": confirmatory_data["C1_correct_compact_time"],
                            label: confirmatory_data[label],
                        },
                        c_grid=c_grid,
                        random_seed=logistic_seed,
                        model_role="confirmatory",
                    )
                    confirmatory_scopes[scope] = scope_report
                    confirmatory_predictions.extend(predictions)
        c3_estimates = [float(item["estimate"]) for item in c3_contrasts]
        task_reports[task] = {
            "primary_scopes": scope_reports,
            "B3_code_count": b3,
            "B4_training_prevalence": b4,
            "paired_patient_bootstrap_contrasts": contrasts,
            "c3_seed_contrasts": c3_contrasts,
            "c3_seed_variability": {
                "seeds": list(spec.c3_seeds),
                "mean_C1_minus_C3_auroc": float(np.mean(c3_estimates)),
                "standard_deviation": float(np.std(c3_estimates, ddof=1)),
            },
            "confirmatory_8b": {
                "selection_basis": "frozen runtime/memory feasibility plan only",
                "selected_priorities": list(selected_priorities),
                "scopes": confirmatory_scopes,
            },
        }
        task_predictions.extend(confirmatory_predictions)
        all_predictions.extend(task_predictions)

    prediction_keys = [
        (
            row["model_role"],
            row["task"],
            row["scope"],
            row["representation"],
            row["perturbation_seed"],
            row["instance_key"],
        )
        for row in all_predictions
    ]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise ContractError("Final predictions are not unique by model/task/scope/condition/patient.")
    predictions_path = spec.private_root / "fit" / "validation_predictions.parquet"
    _atomic_parquet(predictions_path, all_predictions)
    macro_contrast_specs = [
        ("complete_cohort", "C1_correct_compact_time", "C0_order_only"),
        ("c2_perturbable", "C1_correct_compact_time", "C2_reversed_earlier_gaps"),
        ("b1_aligned", "C1_correct_compact_time", "B1_latest_window_only"),
        ("b2_aligned", "C1_correct_compact_time", "B2_latest_code_occurrence"),
        *[
            (
                f"c3_seed_{seed}",
                "C1_correct_compact_time",
                f"C3_shuffled_earlier_time_labels:seed={seed}",
            )
            for seed in spec.c3_seeds
        ],
    ]
    macro_contrasts = [
        macro_paired_contrast(
            all_predictions,
            tasks=spec.tasks,
            scope=scope,
            first=first,
            second=second,
            replicates=replicates,
            seed=bootstrap_seed + 1000 + index,
        )
        for index, (scope, first, second) in enumerate(macro_contrast_specs)
    ]
    elapsed = time.perf_counter() - started
    report = {
        "status": "PASS",
        "protocol_decision": "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "source_git_commit": source_git_commit,
        "scientific_scope": {
            "tasks": list(spec.tasks),
            "splits": list(spec.splits),
            "primary_model": spec.primary_model_repository,
            "confirmatory_model": spec.confirmatory_model_repository,
            "token_budget": spec.token_budget,
            "approach": "M1 direct serialization only",
            "baselines": ["B1", "B2", "B3", "B4"],
            "conditions": ["C0", "C1", "C2", "C3"],
            "c3_seeds": list(spec.c3_seeds),
        },
        "primary_model_identity": {
            "repository": primary_identity.repository,
            "revision_hash": stable_hash(primary_identity.revision),
            "artifact_hash": primary_identity.artifact_hash,
            "tokenizer_hash": primary_identity.tokenizer_hash,
            "frozen": True,
            "full_dimension": True,
            "l2_normalized_float32": True,
        },
        "confirmatory_model_identity": {
            "repository": confirmatory_identity.repository,
            "revision_hash": stable_hash(confirmatory_identity.revision),
            "artifact_hash": confirmatory_identity.artifact_hash,
            "tokenizer_hash": confirmatory_identity.tokenizer_hash,
            "selection_used_outcome_performance": False,
        },
        "foundation": foundation,
        "throughput": throughput,
        "embedding_tasks": list(embedding_reports.values()),
        "tasks": task_reports,
        "equal_weight_panel_macro": _panel_macro(task_reports, spec),
        "equal_weight_panel_macro_paired_bootstrap_contrasts": macro_contrasts,
        "private_predictions_sha256": file_sha256(predictions_path),
        "private_prediction_rows": len(all_predictions),
        "runtime_seconds": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "private_storage_bytes": sum(
            path.stat().st_size for path in spec.private_root.rglob("*") if path.is_file()
        ),
        "content_matching_passed": True,
        "label_index_only_perturbations_passed": True,
        "patient_window_metadata_alignment_passed": True,
        "pilot_artifacts_reused": False,
        "test_predictions_or_performance_used": False,
        "test_results_reported": False,
        "m2_run": False,
        "m3_run": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    _write_immutable_json(spec.aggregate_root / "final_m1_validation.json", report, aggregate=True)
    return report
