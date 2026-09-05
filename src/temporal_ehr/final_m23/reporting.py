"""Prompt 4 validation aggregation, approach selection, and immutable freeze."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import load_final_m1_authorization
from temporal_ehr.final_m1.embedding import condition_slug
from temporal_ehr.final_m1.report import fit_condition_scope, paired_contrast
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.final_m23.m2 import (
    M2_SOURCE_COMPACTION_SCHEMA,
    SummaryCache,
    _write_parquet_once,
    audit_summary_fidelity,
    m2_decoding_variants,
    m2_embedding_root,
    m2_source_text,
    m2_stage_rows,
)
from temporal_ehr.final_m23.m3 import load_m3_sequences
from temporal_ehr.final_m23.protocol import FinalM23Authorization, load_final_m23_authorization
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.pilot.evaluation import (
    calibration_summary,
    fit_logistic_readout,
    metric_set,
    paired_bootstrap_interval,
)
from temporal_ehr.serialization.integrity import stable_hash


def _write_aggregate_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8")) != dict(payload):
            raise ContractError(f"Refusing to overwrite a different aggregate: {path.name}")
    else:
        path.write_text(text, encoding="utf-8")


def _macro(task_metrics: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    required = ("auroc", "auprc", "brier_score", "log_loss")
    return {
        metric: float(np.mean([float(values[metric]) for values in task_metrics.values()]))
        for metric in required
    }


def _load_m2_condition(
    spec: FinalM23Authorization,
    *,
    task: str,
    label: str,
    source_git_commit: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    root = m2_embedding_root(
        spec,
        task=task,
        label=label,
        source_git_commit=source_git_commit,
    )
    metadata_path = root / "patient_metadata.parquet"
    complete_path = root / "COMPLETE.json"
    vectors_path = root / "embeddings.npy"
    if not metadata_path.is_file() or not complete_path.is_file() or not vectors_path.is_file():
        raise ContractError(f"M2 embedding cache is incomplete for {task}/{label}.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("embeddings_sha256") != file_sha256(vectors_path):
        raise ContractError("M2 complete embedding cache hash drifted.")
    metadata = pq.read_table(metadata_path).to_pylist()
    metadata.sort(key=lambda row: int(row["embedding_row_index"]))
    if [int(row["embedding_row_index"]) for row in metadata] != list(range(len(metadata))):
        raise ContractError("M2 patient metadata does not mirror embedding rows.")
    vectors = np.load(vectors_path, mmap_mode="r")
    if len(metadata) != len(vectors):
        raise ContractError("M2 metadata and embedding row counts differ.")
    return metadata, vectors


def _m2_fidelity_records(
    spec: FinalM23Authorization,
    *,
    repository_root: Path,
) -> list[dict[str, Any]]:
    m1 = load_final_m1_authorization(spec.m1_config_path, repository_root=repository_root)
    labels = (
        str(spec.m2["reference_condition"]),
        f"{spec.m2['fidelity_condition']}:seed={spec.m2['fidelity_seed']}",
    )
    records: list[dict[str, Any]] = []
    prompt_hash = stable_hash(
        {"version": spec.m2["prompt_version"], "prompt": spec.m2["prompt"]}
    )
    decoding_hashes = {
        stable_hash(decoding) for _, decoding in m2_decoding_variants(spec.m2)
    }
    for task in spec.tasks:
        prepared = m2_stage_rows(read_final_task_texts(m1, task), spec=spec, stage="reference")
        task_report_path = spec.aggregate_root / "m2" / "tasks" / f"{task}_reference.json"
        if not task_report_path.is_file():
            task_report_path = (
                spec.aggregate_root / "m2" / "tasks" / f"{task}_reference_summaries.json"
            )
        task_report = json.loads(task_report_path.read_text(encoding="utf-8"))
        revision_hash = str(task_report["instruction_model_revision_hash"])
        cache = SummaryCache(spec.private_root / "m2" / "summaries" / task)
        by_source_condition: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for path in (cache.root / "rows").glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("model_revision_hash") == revision_hash
                and payload.get("prompt_hash") == prompt_hash
                and payload.get("decoding_hash") in decoding_hashes
                and payload.get("finish_reason") == "stop"
            ):
                key = (str(payload["source_hash"]), str(payload["condition"]))
                by_source_condition.setdefault(key, []).append(payload)
        for row in prepared:
            label = str(row["representation"])
            seed = row.get("perturbation_seed")
            if seed is not None:
                label = f"{label}:seed={seed}"
            if label not in labels or row["split"] != "validation":
                continue
            source_text = m2_source_text(str(row["text"]), m1.instructions[task])
            source_hash = stable_hash(
                {"schema": M2_SOURCE_COMPACTION_SCHEMA, "source_text": source_text}
            )
            candidates = sorted(
                by_source_condition.get((source_hash, label), []),
                key=lambda value: int(value.get("completion_budget_tokens", 4096)),
            )
            payload = next(
                (
                    value
                    for value in candidates
                    if str(value.get("summary", "")).strip() == source_text.strip()
                ),
                None,
            )
            if payload is None:
                raise ContractError(f"M2 fidelity summary is missing for {task}/{label}.")
            records.append(
                {
                    "task": task,
                    "condition": label,
                    "split": "validation",
                    "instance_key": row["instance_key"],
                    "source_text": source_text,
                    "summary": payload["summary"],
                }
            )
    return records


def fit_m2_validation(
    *,
    repository_root: Path,
    config_path: Path,
    stage: str,
    source_git_commit: str,
) -> dict[str, Any]:
    """Fit M2 with the same logistic procedure and perform required fidelity audit."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    stage_labels = (
        (str(spec.m2["reference_condition"]),)
        if stage == "reference"
        else tuple(str(value) for value in spec.m2["sensitivity_conditions"])
    )
    reference_label = str(spec.m2["reference_condition"])
    fit_labels = stage_labels if stage == "reference" else (reference_label, *stage_labels)
    task_reports: dict[str, Any] = {}
    private_predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for task_index, task in enumerate(spec.tasks):
        condition_data = {
            label: _load_m2_condition(
                spec,
                task=task,
                label=label,
                source_git_commit=source_git_commit,
            )
            for label in fit_labels
        }
        fit_report, predictions = fit_condition_scope(
            task=task,
            scope=f"m2_{stage}",
            condition_data=condition_data,
            c_grid=tuple(float(value) for value in spec.evaluation["regularization_grid"]),
            random_seed=int(spec.evaluation["logistic_seed"]),
            model_role="M2",
        )
        task_reports[task] = fit_report
        if stage == "sensitivity":
            task_reports[task]["paired_patient_bootstrap_contrasts"] = [
                paired_contrast(
                    predictions,
                    task=task,
                    scope=f"m2_{stage}",
                    model_role="M2",
                    first=reference_label,
                    second=label,
                    replicates=int(spec.evaluation["bootstrap_replicates"]),
                    seed=int(spec.evaluation["bootstrap_seed"]) + task_index * 100 + index,
                )
                for index, label in enumerate(stage_labels)
            ]
        private_predictions.extend(predictions)
    prediction_path = spec.private_root / "m2" / "fit" / f"{stage}_validation_predictions.parquet"
    prediction_hash = _write_parquet_once(prediction_path, private_predictions)
    task_metrics = {
        task: task_reports[task]["representations"][reference_label]["metrics"]
        for task in spec.tasks
    }
    macros_by_condition = {
        label: _macro(
            {
                task: task_reports[task]["representations"][label]["metrics"]
                for task in spec.tasks
            }
        )
        for label in fit_labels
    }
    report: dict[str, Any] = {
        "status": "PASS",
        "protocol_decision": f"M2_{stage.upper()}_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "stage": stage,
        "approach": "M2 single-pass on-premise summary embedding",
        "conditions": list(stage_labels),
        "tasks": task_reports,
        "equal_weight_panel_macro": _macro(task_metrics),
        "equal_weight_panel_macro_by_condition": macros_by_condition,
        "private_predictions_sha256": prediction_hash,
        "single_pass_nonrecursive": True,
        "external_api_used": False,
        "same_logistic_procedure_as_m1": True,
        "runtime_seconds": time.perf_counter() - started,
        "test_predictions_or_performance_used": False,
        "test_results_reported": False,
    }
    if stage == "reference":
        fidelity = audit_summary_fidelity(
            _m2_fidelity_records(spec, repository_root=repository_root),
            thresholds=spec.m2["fidelity"],
            histories_per_task_condition=int(spec.m2["fidelity"]["histories_per_task_condition"]),
        )
        manual_path = spec.aggregate_root / "m2_manual_fidelity_review.json"
        if not manual_path.is_file():
            raise ContractError(
                "M2 reference fit requires the completed private 40-history manual fidelity review."
            )
        manual = json.loads(manual_path.read_text(encoding="utf-8"))
        if any(
            (
                manual.get("manual_review_completed") is not True,
                manual.get("validation_only") is not True,
                manual.get("sample_histories") != 40,
                manual.get("task_condition_groups") != 8,
                manual.get("held_out_prompt_refinement") is not False,
            )
        ):
            raise ContractError("The M2 manual fidelity aggregate is incomplete or drifted.")
        report["fidelity_validation"] = fidelity
        report["manual_fidelity_review"] = manual
        report["eligible_for_approach_selection"] = (
            fidelity["status"] == "PASS" and manual.get("status") == "PASS"
        )
    else:
        report["selection_authorization_sha256"] = file_sha256(spec.aggregate_root / "approach_selection.json")
    _write_aggregate_once(spec.aggregate_root / f"m2_{stage}.json", report)
    return report


def _load_m3_seed_predictions(
    spec: FinalM23Authorization,
    *,
    task: str,
    label: str,
    seed: int,
) -> list[dict[str, Any]]:
    root = spec.private_root / "m3" / "fits" / task / condition_slug(label) / f"seed_{seed}"
    complete_path = root / "COMPLETE.json"
    predictions_path = root / "validation_predictions.parquet"
    if not complete_path.is_file() or not predictions_path.is_file():
        raise ContractError(f"M3 fit is incomplete for {task}/{label}/seed={seed}.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("predictions_sha256") != file_sha256(predictions_path):
        raise ContractError("M3 seed prediction hash drifted.")
    return pq.read_table(predictions_path).to_pylist()


def _average_seed_predictions(
    rows_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    mappings = {
        seed: {str(row["instance_key"]): dict(row) for row in rows}
        for seed, rows in rows_by_seed.items()
    }
    key_sets = [set(rows) for rows in mappings.values()]
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ContractError("M3 seeds do not contain identical validation patients.")
    output = []
    for key in sorted(key_sets[0]):
        targets = {bool(rows[key]["target"]) for rows in mappings.values()}
        if len(targets) != 1:
            raise ContractError("M3 seed targets are not aligned.")
        first = next(iter(mappings.values()))[key]
        output.append(
            {
                "task": first["task"],
                "condition": first["condition"],
                "instance_key": key,
                "target": targets.pop(),
                "probability": float(np.mean([float(rows[key]["probability"]) for rows in mappings.values()])),
                "seeds_averaged": sorted(mappings),
            }
        )
    return output


def _mean_pool_baseline(
    spec: FinalM23Authorization,
    *,
    task: str,
    label: str,
) -> dict[str, Any]:
    metadata, sequences = load_m3_sequences(spec, task=task, label=label)
    train = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    validation = [index for index, row in enumerate(metadata) if row["split"] == "validation"]
    train_x = np.asarray([np.mean(sequences[index], axis=0) for index in train], dtype=np.float32)
    validation_x = np.asarray([np.mean(sequences[index], axis=0) for index in validation], dtype=np.float32)
    train_y = np.asarray([metadata[index]["target"] for index in train], dtype=np.int8)
    validation_y = np.asarray([metadata[index]["target"] for index in validation], dtype=np.int8)
    _, report = fit_logistic_readout(
        train_x,
        train_y,
        validation_x,
        validation_y,
        c_grid=tuple(float(value) for value in spec.evaluation["regularization_grid"]),
        random_seed=int(spec.evaluation["logistic_seed"]),
    )
    return report


def select_approach(
    evidence: Mapping[str, Mapping[str, Any]],
    *,
    minimum_margin: float,
    no_margin_priority: Sequence[str],
) -> dict[str, Any]:
    """Apply the locked AUROC margin; log loss is only a documented secondary tie-breaker."""
    eligible = {
        approach: values
        for approach, values in evidence.items()
        if values.get("eligible", True) is True
    }
    if "M1" not in eligible or len(eligible) < 2:
        raise ContractError("Approach selection requires M1 and at least one additional eligible reference.")
    ranked = sorted(
        eligible,
        key=lambda name: (
            -float(eligible[name]["macro_auroc"]),
            float(eligible[name]["macro_log_loss"]),
            list(no_margin_priority).index(name),
        ),
    )
    top = ranked[0]
    runner_up = ranked[1]
    margin = float(eligible[top]["macro_auroc"]) - float(eligible[runner_up]["macro_auroc"])
    if margin >= minimum_margin:
        selected = top
        reason = "top validation macro-AUROC exceeded every eligible alternative by at least 0.005"
    else:
        selected = next(name for name in no_margin_priority if name in eligible)
        reason = "no eligible approach cleared the 0.005 margin; applied frozen M1-then-M2-then-M3 priority"
    return {
        "selected_approach": selected,
        "selection_reason": reason,
        "observed_top_margin": margin,
        "minimum_margin": minimum_margin,
        "primary_metric": "equal-weight validation macro-AUROC",
        "second_tiebreaker": "validation log loss",
        "no_margin_priority": list(no_margin_priority),
        "ranked_eligible_approaches": ranked,
    }


def aggregate_m3_validation(
    *,
    repository_root: Path,
    config_path: Path,
    stage: str,
    source_git_commit: str,
    evidence_source_git_commit: str | None = None,
    recovery_report_path: Path | None = None,
    recovery_report_sha256: str | None = None,
) -> dict[str, Any]:
    """Average three GRU seeds, report mean-pool baseline, and select after C1 references."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    evidence_commit = evidence_source_git_commit or source_git_commit
    recovery_provenance = None
    if evidence_commit != source_git_commit:
        expected_root = spec.aggregate_root.resolve()
        if (
            stage != "reference"
            or recovery_report_path is None
            or not recovery_report_path.resolve().is_relative_to(expected_root)
            or not recovery_report_path.is_file()
            or recovery_report_sha256 is None
            or file_sha256(recovery_report_path) != recovery_report_sha256
        ):
            raise ContractError("Cross-commit M3 reuse requires the exact hashed recovery report.")
        recovery_provenance = json.loads(recovery_report_path.read_text(encoding="utf-8"))
        if (
            recovery_provenance.get("status") != "PASS"
            or recovery_provenance.get("source_git_commit") != source_git_commit
            or recovery_provenance.get("m3_compute_git_commit") != evidence_commit
            or recovery_provenance.get("m3_rerun_required") is not False
            or recovery_provenance.get("thinking_disabled") is not True
            or recovery_provenance.get("test_predictions_or_performance_used") is not False
        ):
            raise ContractError("Cross-commit M3 recovery report provenance drifted.")
    stage_labels = (
        (str(spec.m3["reference_condition"]),)
        if stage == "reference"
        else tuple(str(value) for value in spec.m3["sensitivity_conditions"])
    )
    reference_label = str(spec.m3["reference_condition"])
    report_labels = stage_labels if stage == "reference" else (reference_label, *stage_labels)
    seeds = tuple(int(value) for value in spec.m3["training_seeds"])
    task_reports: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for task_index, task in enumerate(spec.tasks):
        conditions: dict[str, Any] = {}
        averaged_by_label: dict[str, list[dict[str, Any]]] = {}
        preparation_hashes = set()
        for seed in seeds:
            task_seed_path = (
                spec.aggregate_root / "m3" / "tasks" / f"{task}_seed_{seed}_{stage}.json"
            )
            if not task_seed_path.is_file():
                raise ContractError("M3 aggregate requires every task/seed execution report.")
            task_seed_report = json.loads(task_seed_path.read_text(encoding="utf-8"))
            if (
                task_seed_report.get("status") != "PASS"
                or task_seed_report.get("source_git_commit") != evidence_commit
                or task_seed_report.get("task") != task
                or task_seed_report.get("seed") != seed
                or task_seed_report.get("stage") != stage
                or task_seed_report.get("seed_independent_cache_prepared_once") is not True
                or task_seed_report.get("test_split_used") is not False
            ):
                raise ContractError("M3 task/seed execution provenance drifted.")
            preparation_hashes.add(task_seed_report.get("preparation_report_sha256"))
        if len(preparation_hashes) != 1 or None in preparation_hashes:
            raise ContractError("M3 seeds did not share one immutable task preparation report.")
        for label in report_labels:
            rows_by_seed = {
                seed: _load_m3_seed_predictions(spec, task=task, label=label, seed=seed)
                for seed in seeds
            }
            averaged = _average_seed_predictions(rows_by_seed)
            averaged_by_label[label] = averaged
            all_predictions.extend(averaged)
            targets = np.asarray([row["target"] for row in averaged], dtype=np.int8)
            probabilities = np.asarray([row["probability"] for row in averaged], dtype=np.float64)
            seed_metrics = {
                str(seed): metric_set(
                    np.asarray([row["target"] for row in rows_by_seed[seed]], dtype=np.int8),
                    np.asarray([row["probability"] for row in rows_by_seed[seed]], dtype=np.float64),
                )
                for seed in seeds
            }
            conditions[label] = {
                "averaged_seed_metrics": metric_set(targets, probabilities),
                "calibration": calibration_summary(targets, probabilities),
                "seed_metrics": seed_metrics,
                "seed_auroc_standard_deviation": float(
                    np.std([values["auroc"] for values in seed_metrics.values()], ddof=1)
                ),
                "three_seed_probability_average": True,
            }
        task_reports[task] = {
            "conditions": conditions,
            "seed_independent_preparation_report_sha256": preparation_hashes.pop(),
        }
        if stage == "reference":
            task_reports[task]["mean_pool_logistic_baseline"] = _mean_pool_baseline(
                spec,
                task=task,
                label=reference_label,
            )
        else:
            reference_rows = averaged_by_label[reference_label]
            reference_map = {str(row["instance_key"]): row for row in reference_rows}
            contrasts = []
            for index, label in enumerate(stage_labels):
                comparison_map = {
                    str(row["instance_key"]): row for row in averaged_by_label[label]
                }
                keys = sorted(set(reference_map) & set(comparison_map))
                if not keys:
                    raise ContractError("M3 sensitivity contrast has no aligned validation patients.")
                targets = np.asarray([reference_map[key]["target"] for key in keys], dtype=np.int8)
                if targets.tolist() != [comparison_map[key]["target"] for key in keys]:
                    raise ContractError("M3 sensitivity targets drifted across conditions.")
                contrasts.append(
                    {
                        "first": reference_label,
                        "second": label,
                        "instances": len(keys),
                        **paired_bootstrap_interval(
                            targets,
                            np.asarray([reference_map[key]["probability"] for key in keys]),
                            np.asarray([comparison_map[key]["probability"] for key in keys]),
                            replicates=int(spec.evaluation["bootstrap_replicates"]),
                            seed=int(spec.evaluation["bootstrap_seed"]) + task_index * 100 + index,
                        ),
                    }
                )
            task_reports[task]["paired_patient_bootstrap_contrasts"] = contrasts
    predictions_path = spec.private_root / "m3" / "fit" / f"{stage}_averaged_predictions.parquet"
    predictions_hash = _write_parquet_once(predictions_path, all_predictions)
    reference_metrics = {
        task: task_reports[task]["conditions"][reference_label]["averaged_seed_metrics"]
        for task in spec.tasks
    }
    macros_by_condition = {
        label: _macro(
            {
                task: task_reports[task]["conditions"][label]["averaged_seed_metrics"]
                for task in spec.tasks
            }
        )
        for label in report_labels
    }
    report: dict[str, Any] = {
        "status": "PASS",
        "protocol_decision": f"M3_{stage.upper()}_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "m3_compute_git_commit": evidence_commit,
        "validated_cross_commit_reuse": evidence_commit != source_git_commit,
        "m3_recovery_report_sha256": (
            recovery_report_sha256 if recovery_provenance is not None else None
        ),
        "stage": stage,
        "approach": "M3 frozen per-window Qwen plus one-layer unidirectional GRU",
        "conditions": list(stage_labels),
        "training_seeds": list(seeds),
        "tasks": task_reports,
        "equal_weight_panel_macro": _macro(reference_metrics),
        "equal_weight_panel_macro_by_condition": macros_by_condition,
        "private_averaged_predictions_sha256": predictions_hash,
        "mean_pool_logistic_baseline_reported": stage == "reference",
        "patient_alignment_passed": True,
        "test_predictions_or_performance_used": False,
        "test_results_reported": False,
    }
    _write_aggregate_once(spec.aggregate_root / f"m3_{stage}.json", report)
    if stage == "reference":
        m1 = json.loads(spec.m1_report_path.read_text(encoding="utf-8"))
        m2 = json.loads((spec.aggregate_root / "m2_reference.json").read_text(encoding="utf-8"))
        m1_metrics = m1["equal_weight_panel_macro"]["C1_complete_cohort"]["metrics"]
        m2_metrics = m2["equal_weight_panel_macro"]
        m3_metrics = report["equal_weight_panel_macro"]
        evidence = {
            "M1": {
                "macro_auroc": m1_metrics["auroc"],
                "macro_log_loss": m1_metrics["log_loss"],
                "eligible": True,
            },
            "M2": {
                "macro_auroc": m2_metrics["auroc"],
                "macro_log_loss": m2_metrics["log_loss"],
                "eligible": m2.get("eligible_for_approach_selection") is True,
            },
            "M3": {
                "macro_auroc": m3_metrics["auroc"],
                "macro_log_loss": m3_metrics["log_loss"],
                "eligible": True,
            },
        }
        rule = spec.evaluation["approach_selection"]
        selection = {
            "status": "PASS",
            "protocol_decision": "PROMPT4_REFERENCE_SELECTION_COMPLETE_TEST_REMAINS_LOCKED",
            "experiment_id": spec.experiment_id,
            "source_git_commit": source_git_commit,
            "reference_condition": "C1_correct_compact_time",
            "reference_cells_passed": True,
            "evidence": evidence,
            **select_approach(
                evidence,
                minimum_margin=float(rule["minimum_margin"]),
                no_margin_priority=tuple(rule["no_margin_priority"]),
            ),
            "outcome_performance_source": "training/validation only",
            "test_predictions_or_performance_used": False,
            "test_results_reported": False,
        }
        _write_aggregate_once(spec.aggregate_root / "approach_selection.json", selection)
    return report


def freeze_prompt4_manifest(
    *,
    repository_root: Path,
    config_path: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Freeze the selected approach and authorized contrasts without reading held-out data."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    selection_path = spec.aggregate_root / "approach_selection.json"
    if not selection_path.is_file():
        raise ContractError("Prompt 4 freeze requires the completed reference approach selection.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = str(selection.get("selected_approach"))
    required_reports = [
        spec.m1_report_path,
        spec.prompt3_handoff_path,
        spec.aggregate_root / "m2_reference.json",
        spec.aggregate_root / "m3_reference.json",
        selection_path,
    ]
    if selected == "M2":
        required_reports.append(spec.aggregate_root / "m2_sensitivity.json")
    elif selected == "M3":
        required_reports.append(spec.aggregate_root / "m3_sensitivity.json")
    elif selected != "M1":
        raise ContractError("Prompt 4 selected approach is not M1, M2, or M3.")
    if any(not path.is_file() for path in required_reports):
        raise ContractError("Prompt 4 freeze is missing a required reference or sensitivity report.")
    manifest_body = {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "experiment_config_path": str(config_path.relative_to(repository_root)),
        "experiment_config_sha256": file_sha256(config_path),
        "selected_approach": selected,
        "authorized_one_time_test_approach": selected,
        "authorized_one_time_test_contrasts": [
            "C1_correct_compact_time minus B1_latest_window_only",
            "C1_correct_compact_time minus each of three fixed C3 shuffled-label seeds",
            "C1_correct_compact_time minus C0_order_only",
            "C1_correct_compact_time minus C2_reversed_earlier_gaps",
        ],
        "tasks": list(spec.tasks),
        "splits_used_for_development": ["train", "validation"],
        "test_split_accessed": False,
        "reports": {
            str(path.relative_to(repository_root)): file_sha256(path)
            for path in required_reports
        },
        "selection_rule": spec.evaluation["approach_selection"],
        "m2_prompt_hash": stable_hash(
            {"version": spec.m2["prompt_version"], "prompt": spec.m2["prompt"]}
        ),
        "m2_decoding_hash": stable_hash(spec.m2["decoding"]),
        "m2_completion_policy_hash": stable_hash(
            {name: decoding for name, decoding in m2_decoding_variants(spec.m2)}
        ),
        "m3_architecture_hash": stable_hash(spec.m3),
        "protocol_frozen": True,
        "one_time_held_out_run_authorized_by_prompt4": True,
        "test_predictions_or_performance_used": False,
    }
    manifest = {**manifest_body, "frozen_manifest_sha256": stable_hash(manifest_body)}
    _write_aggregate_once(spec.aggregate_root / "FROZEN_EXPERIMENT_MANIFEST.json", manifest)
    return {"status": "PASS", **manifest}
