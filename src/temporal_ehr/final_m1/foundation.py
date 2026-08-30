"""DGX-only construction and semantic validation of the locked final M1 foundation."""

from __future__ import annotations

import json
import os
import resource
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.history import iter_strict_histories
from temporal_ehr.experiments import FinalM1Authorization, load_final_m1_authorization, window_days_for_task
from temporal_ehr.features.count_baseline import run_count_baseline
from temporal_ehr.final_m1.cohort import build_final_cohort
from temporal_ehr.final_m1.gates import require_prompt3_hard_gates
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.pilot.index import finalize_indices, prepare_instance
from temporal_ehr.pilot.model import load_model_identity
from temporal_ehr.serialization.blocks import deduplicate_events, event_from_mapping
from temporal_ehr.serialization.integrity import stable_hash
from temporal_ehr.serialization.packing import load_qwen_tokenizer
from temporal_ehr.serialization.render import load_code_descriptions


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ContractError(f"Refusing to write empty final private Parquet: {path.name}")
    if path.exists():
        raise ContractError(f"Refusing to overwrite existing final private Parquet: {path.name}")
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


def _write_immutable_json(path: Path, payload: Mapping[str, Any], *, aggregate: bool = False) -> None:
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if aggregate:
        assert_privacy_safe_report(text)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ContractError(f"Refusing to overwrite a different final artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _numeric_summary(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "median": 0.0, "mean": 0.0, "maximum": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
    }


def _representation_label(row: Mapping[str, Any]) -> str:
    seed = row.get("perturbation_seed")
    return str(row["representation"]) if seed is None else f"{row['representation']}:seed={seed}"


def _preparation_coverage(
    indices: Sequence[Mapping[str, Any]],
    texts: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_representation: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in indices:
        by_representation[str(row["representation_key"])].append(row)
    text_by_key = {str(row["representation_key"]): row for row in texts}
    # Empty-history C0/C1 rows have no source-window rows, so seed them directly.
    for key in text_by_key:
        by_representation.setdefault(key, [])
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for representation_key, rows in by_representation.items():
        text = text_by_key.get(representation_key)
        if rows:
            label = _representation_label(rows[0])
            eligible = any(row["retention_status"] != "INELIGIBLE" for row in rows)
        elif text is not None:
            label = _representation_label(text)
            eligible = True
        else:
            raise ContractError("Patient-window index has an orphan representation key.")
        if eligible != (text is not None):
            raise ContractError("Patient-window eligibility does not align with final embedding metadata.")
        retained = [row for row in rows if row["retention_status"] in {"RETAINED", "PARTIAL"}]
        grouped[label].append(
            {
                "eligible": eligible,
                "source_windows": len(rows),
                "retained_windows": len(retained),
                "source_events": sum(int(row["event_count"]) for row in rows),
                "retained_events": sum(int(row["retained_event_count"]) for row in retained),
                "token_count": int(text["token_count"]) if text is not None else None,
                "truncated": bool(text["truncated"]) if text is not None else None,
                "retained_historical_span_days": (
                    float(text["retained_historical_span_days"]) if text is not None else None
                ),
            }
        )
    output: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(grouped.items()):
        eligible = [row for row in rows if row["eligible"]]
        output[label] = {
            "source_histories": len(rows),
            "eligible_histories": len(eligible),
            "ineligible_histories": len(rows) - len(eligible),
            "source_windows": sum(row["source_windows"] for row in eligible),
            "retained_windows": sum(row["retained_windows"] for row in eligible),
            "source_events": sum(row["source_events"] for row in eligible),
            "retained_events": sum(row["retained_events"] for row in eligible),
            "truncated_histories": sum(bool(row["truncated"]) for row in eligible),
        }
        if eligible:
            output[label]["token_count"] = _numeric_summary([int(row["token_count"]) for row in eligible])
            output[label]["retained_historical_span_days"] = _numeric_summary(
                [float(row["retained_historical_span_days"]) for row in eligible]
            )
    return output


def _assert_content_and_label_only_perturbations(
    indices: Sequence[Mapping[str, Any]],
    texts: Sequence[Mapping[str, Any]],
) -> None:
    full = {
        "C0_order_only",
        "C1_correct_compact_time",
        "C2_reversed_earlier_gaps",
        "C3_shuffled_earlier_time_labels",
    }
    by_instance: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in texts:
        if row["representation"] in full:
            by_instance[str(row["instance_key"])].append(row)
    if not by_instance:
        raise ContractError("No final C0-C3 inputs were prepared.")
    for rows in by_instance.values():
        if len({row["source_content_hash"] for row in rows}) != 1:
            raise ContractError("C0-C3 source clinical event arrays differ.")
        if len({row["retained_content_hash"] for row in rows}) != 1:
            raise ContractError("C0-C3 retained clinical event arrays differ.")
    by_instance_windows: defaultdict[tuple[str, int | None], list[Mapping[str, Any]]] = defaultdict(list)
    for row in indices:
        if row["representation"] in full:
            by_instance_windows[(str(row["instance_key"]), row.get("perturbation_seed"))].append(row)
    # Every altered label still points at the unchanged source-window content hash.
    for rows in by_instance_windows.values():
        by_window: defaultdict[int, set[str]] = defaultdict(set)
        for row in rows:
            by_window[int(row["source_window_index"])].add(str(row["deduplicated_clinical_content_hash"]))
        if any(len(values) != 1 for values in by_window.values()):
            raise ContractError("Temporal perturbation rearranged a clinical event array.")


def perturbation_strength(indices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate how many label indices changed without exposing patient/window rows."""
    grouped: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in indices:
        if row["representation"] in {"C2_reversed_earlier_gaps", "C3_shuffled_earlier_time_labels"}:
            grouped[str(row["representation_key"])].append(row)
    by_condition: defaultdict[str, list[dict[str, float]]] = defaultdict(list)
    for rows in grouped.values():
        first = rows[0]
        label = _representation_label(first)
        changed = [
            abs(int(row["displayed_time_label_index"]) - int(row["correct_time_label_index"]))
            for row in rows
            if row["displayed_time_label_index"] is not None
            and row["displayed_time_label_index"] != row["correct_time_label_index"]
        ]
        if first["retention_status"] != "INELIGIBLE" and not changed:
            raise ContractError("Eligible C2/C3 representation has no changed label index.")
        by_condition[label].append(
            {
                "changed_labels": float(len(changed)),
                "mean_absolute_window_shift": float(np.mean(changed)) if changed else 0.0,
                "maximum_absolute_window_shift": float(max(changed)) if changed else 0.0,
            }
        )
    return {
        label: {
            "eligible_histories": sum(row["changed_labels"] > 0 for row in rows),
            "changed_label_count": int(sum(row["changed_labels"] for row in rows)),
            "mean_changed_labels_per_eligible_history": (
                float(np.mean([row["changed_labels"] for row in rows if row["changed_labels"] > 0]))
                if any(row["changed_labels"] > 0 for row in rows)
                else 0.0
            ),
            "mean_absolute_window_shift": (
                float(np.mean([row["mean_absolute_window_shift"] for row in rows if row["changed_labels"] > 0]))
                if any(row["changed_labels"] > 0 for row in rows)
                else 0.0
            ),
            "maximum_absolute_window_shift": float(max(row["maximum_absolute_window_shift"] for row in rows)),
        }
        for label, rows in sorted(by_condition.items())
    }


def validate_outcome_contracts(spec: FinalM1Authorization, registry: Mapping[str, Any]) -> dict[str, Any]:
    """Validate clinical target semantics without consulting model performance."""
    expected = {
        "lab_anemia": {
            "question_fragment": "next qualifying hemoglobin result",
            "family": "laboratory_result",
            "horizon": "next qualifying result",
            "shortcut": "incident anemia diagnosis",
        },
        "guo_readmission": {
            "question_fragment": "readmitted within 30 days",
            "family": "operational_outcome",
            "horizon": "30 days",
            "shortcut": "at or after prediction time",
        },
        "new_hypertension": {
            "question_fragment": "first hypertension diagnosis occur within one year",
            "family": "incident_diagnosis",
            "horizon": "365 days",
            "shortcut": "prevalent hypertension",
        },
        "chexpert": {
            "question_fragment": "chest radiograph contain a finding",
            "family": "imaging_finding",
            "horizon": "official CheXpert finding label",
            "shortcut": "at or after prediction time",
        },
    }
    audits = {}
    for task, rule in expected.items():
        configured = registry["tasks"][task]
        contract = spec.raw["outcome_contracts"][task]
        if (
            configured.get("family") != rule["family"]
            or rule["question_fragment"] not in str(configured.get("prediction_question", ""))
            or contract.get("horizon") != rule["horizon"]
            or rule["shortcut"] not in str(contract.get("not_outcome", ""))
        ):
            raise ContractError(f"Outcome/shortcut definition drifted for {task}.")
        if task == "chexpert" and "24 hours before" not in str(configured.get("timing", "")):
            raise ContractError("CheXpert prediction no longer precedes the report/finding.")
        audits[task] = {
            "status": "PASS",
            "official_outcome_definition": str(contract["outcome"]),
            "official_prediction_timing": str(contract["official_timing"]),
            "outcome_horizon": str(contract["horizon"]),
            "shortcut_excluded": str(contract["not_outcome"]),
            "official_prediction_time_used": True,
            "all_retained_events_strictly_before_prediction_time": True,
            "target_defining_future_events_retained": 0,
            "prior_related_predictors_before_prediction_time_allowed": True,
        }
    return audits


def _metadata_rows(texts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One private metadata row for every prepared primary embedding row/condition."""
    output: list[dict[str, Any]] = []
    for row in texts:
        metadata = {key: value for key, value in row.items() if key != "text"}
        metadata["model_role"] = "primary"
        metadata["mirrors_embedding_row"] = True
        output.append(metadata)
    if len(output) != len(texts) or len({row["cache_key"] for row in output}) != len(texts):
        raise ContractError("Final patient-window metadata does not mirror prepared embedding rows.")
    return output


def _task_prepare_summary(
    *,
    task: str,
    indices: Sequence[Mapping[str, Any]],
    texts: Sequence[Mapping[str, Any]],
    source_counts: Sequence[int],
    deduplicated_counts: Sequence[int],
    recurrence_counts: Sequence[int],
) -> dict[str, Any]:
    representations: defaultdict[str, int] = defaultdict(int)
    for row in texts:
        representations[_representation_label(row)] += 1
    c1 = [row for row in texts if row["representation"] == "C1_correct_compact_time"]
    return {
        "task": task,
        "source_instances": len(source_counts),
        "patient_window_index_rows": len(indices),
        "patient_window_metadata_rows": len(texts),
        "eligible_embedding_rows": len(texts),
        "representations": dict(sorted(representations.items())),
        "source_event_count": _numeric_summary(source_counts),
        "deduplicated_event_count": _numeric_summary(deduplicated_counts),
        "exact_duplicate_rows_removed": int(sum(source_counts) - sum(deduplicated_counts)),
        "recurrent_event_count_after_exact_deduplication": int(sum(recurrence_counts)),
        "histories_with_recurrence": int(sum(value > 0 for value in recurrence_counts)),
        "token_count": _numeric_summary([int(row["token_count"]) for row in texts]),
        "truncated_embedding_rows": int(sum(bool(row["truncated"]) for row in texts)),
        "source_historical_span_days": _numeric_summary(
            [float(row["source_historical_span_days"]) for row in c1]
        ),
        "representation_coverage": _preparation_coverage(indices, texts),
        "perturbation_strength": perturbation_strength(indices),
        "content_matching_passed": True,
        "clinical_event_arrays_rearranged_by_c2_or_c3": False,
        "metadata_mirrors_every_embedding_row": True,
        "test_split_used": False,
    }


def build_final_foundation(
    *,
    repository_root: Path,
    config_path: Path,
    event_path: Path,
    asset_root: Path,
    model_root: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Build the final cohort, inputs, indices, audits, and B3 baseline; never run a model."""
    started = time.perf_counter()
    root = repository_root.resolve()
    spec = load_final_m1_authorization(config_path, repository_root=root)
    hard_gates = require_prompt3_hard_gates(root, config_path)
    if len(source_git_commit) != 40 or any(character not in "0123456789abcdef" for character in source_git_commit):
        raise ContractError("Final foundation requires the full source Git commit from the Slurm wrapper.")
    actual_event_hash = file_sha256(event_path)
    if actual_event_hash != spec.raw["dataset_release"]["event_data_sha256"]:
        raise ContractError("Governed event data hash does not match the pinned EHRSHOT release.")
    registry = yaml.safe_load((root / "configs/tasks.yaml").read_text(encoding="utf-8"))
    outcome_audits = validate_outcome_contracts(spec, registry)
    cohort_summary = build_final_cohort(asset_root=asset_root, registry=registry, spec=spec)
    cohort_complete = json.loads((spec.cohort_path.parent / "COMPLETE.json").read_text(encoding="utf-8"))
    cohort_hash = str(cohort_complete["cohort_sha256"])

    identity = load_model_identity(
        model_root / spec.primary_model_directory_name,
        expected_repository=spec.primary_model_repository,
    )
    tokenizer = load_qwen_tokenizer(identity.directory)
    descriptions = load_code_descriptions(asset_root)
    descriptions_hash = stable_hash(descriptions)
    config_hash = stable_hash(spec.raw)
    task_summaries: list[dict[str, Any]] = []
    for task in spec.tasks:
        task_root = spec.private_root / "prepared" / task
        complete_path = task_root / "COMPLETE.json"
        task_input_hash = stable_hash(
            {
                "experiment_id": spec.experiment_id,
                "experiment_config_hash": config_hash,
                "cohort_sha256": cohort_hash,
                "event_data_sha256": actual_event_hash,
                "descriptions_hash": descriptions_hash,
                "model_artifact_hash": identity.artifact_hash,
                "tokenizer_hash": identity.tokenizer_hash,
                "source_git_commit": source_git_commit,
                "task": task,
            }
        )
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            paths = {
                "patient_window_index_sha256": task_root / "patient_window_index.parquet",
                "patient_window_metadata_sha256": task_root / "patient_window_metadata.parquet",
                "texts_sha256": task_root / "texts.parquet",
            }
            if complete.get("task_input_hash") != task_input_hash or any(
                not path.is_file() or complete.get(field) != file_sha256(path)
                for field, path in paths.items()
            ):
                raise ContractError(f"Completed final preparation is stale for task {task}.")
            task_summaries.append(dict(complete["aggregate_summary"]))
            continue
        if task_root.exists() and any(task_root.iterdir()):
            raise ContractError(f"Partial final preparation exists for task {task}; refusing overwrite.")
        index_rows: list[dict[str, Any]] = []
        text_rows: list[dict[str, Any]] = []
        source_counts: list[int] = []
        deduplicated_counts: list[int] = []
        recurrence_counts: list[int] = []
        for instance in iter_strict_histories(
            event_path,
            spec.cohort_path,
            tasks=(task,),
            splits=spec.splits,
        ):
            if any(row["time"] >= instance.prediction_time for row in instance.events):
                raise ContractError("Strict history query retained an event at or after prediction time.")
            events = tuple(event_from_mapping(row) for row in instance.events)
            deduplicated, _ = deduplicate_events(events)
            prepared = prepare_instance(
                instance,
                experiment_id=spec.experiment_id,
                window_days=window_days_for_task(task, registry),
                c3_seeds=spec.c3_seeds,
                descriptions=descriptions,
                tokenizer=tokenizer,
                instruction=spec.instructions[task],
                budget=spec.token_budget,
                model_hash=identity.artifact_hash,
                tokenizer_hash=identity.tokenizer_hash,
            )
            index_rows.extend(prepared.index_rows)
            text_rows.extend(prepared.text_rows)
            source_counts.append(len(events))
            deduplicated_counts.append(len(deduplicated))
            recurrence_counts.append(len(deduplicated) - len({event.code for event in deduplicated}))
        indices, texts = finalize_indices(
            index_rows,
            text_rows,
            experiment_id=spec.experiment_id,
            model_hash=identity.artifact_hash,
            tokenizer_hash=identity.tokenizer_hash,
            instructions=spec.instructions,
            budget=spec.token_budget,
        )
        _assert_content_and_label_only_perturbations(indices, texts)
        expected_instances = sum(
            int(group["instances"])
            for group in cohort_summary["groups"]
            if group["task"] == task
        )
        c1_rows = [row for row in texts if row["representation"] == "C1_correct_compact_time"]
        if len(source_counts) != expected_instances or len(c1_rows) != expected_instances:
            raise ContractError("Final C1 preparation is not one-to-one with the final cohort.")
        metadata = _metadata_rows(texts)
        task_root.mkdir(parents=True, exist_ok=True)
        index_path = task_root / "patient_window_index.parquet"
        metadata_path = task_root / "patient_window_metadata.parquet"
        texts_path = task_root / "texts.parquet"
        _atomic_parquet(index_path, indices)
        _atomic_parquet(metadata_path, metadata)
        _atomic_parquet(texts_path, texts)
        summary = _task_prepare_summary(
            task=task,
            indices=indices,
            texts=texts,
            source_counts=source_counts,
            deduplicated_counts=deduplicated_counts,
            recurrence_counts=recurrence_counts,
        )
        complete = {
            "state": "COMPLETE",
            "experiment_id": spec.experiment_id,
            "task": task,
            "task_input_hash": task_input_hash,
            "patient_window_index_sha256": file_sha256(index_path),
            "patient_window_metadata_sha256": file_sha256(metadata_path),
            "texts_sha256": file_sha256(texts_path),
            "aggregate_summary": summary,
        }
        atomic_json(complete_path, complete)
        task_summaries.append(summary)

    count_complete = spec.private_root / "count_baseline" / "COMPLETE.json"
    count_report_path = spec.aggregate_root / "count_baseline" / "validation_results.json"
    if count_complete.is_file():
        payload = json.loads(count_complete.read_text(encoding="utf-8"))
        if (
            payload.get("cohort_sha256") != cohort_hash
            or not count_report_path.is_file()
            or payload.get("report_sha256") != file_sha256(count_report_path)
        ):
            raise ContractError("Existing final B3 completion is stale or hash-mismatched.")
        count_report = json.loads(count_report_path.read_text(encoding="utf-8"))
    else:
        private_count_root = spec.private_root / "count_baseline" / "tasks"
        if private_count_root.exists() and any(private_count_root.iterdir()):
            raise ContractError("Partial final B3 artifacts exist; refusing silent overwrite.")
        count_report = run_count_baseline(
            event_path,
            spec.cohort_path,
            tasks=spec.tasks,
            private_root=private_count_root,
            report_root=count_report_path.parent,
            c_grid=tuple(float(value) for value in spec.raw["regularization_grid"]),
            random_seed=int(spec.raw["logistic_seed"]),
            minimum_report_group_size=int(spec.raw["minimum_report_group_size"]),
        )
        if count_report.get("status") != "PASS" or count_report.get("test_split_inspected") is not False:
            raise ContractError("Final B3 did not pass its train/validation-only gate.")
        atomic_json(
            count_complete,
            {
                "state": "COMPLETE",
                "experiment_id": spec.experiment_id,
                "cohort_sha256": cohort_hash,
                "report_sha256": file_sha256(count_report_path),
            },
        )

    elapsed = time.perf_counter() - started
    storage_bytes = sum(
        path.stat().st_size
        for path in spec.private_root.rglob("*")
        if path.is_file()
    )
    report = {
        "status": "PASS",
        "protocol_decision": "FINAL_FOUNDATION_COMPLETE_M1_AUTHORIZED",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": config_hash,
        "source_git_commit": source_git_commit,
        "hard_gates": hard_gates,
        "scientific_scope": {
            "tasks": list(spec.tasks),
            "splits": list(spec.splits),
            "token_budget": spec.token_budget,
            "window_days_by_task": {
                task: window_days_for_task(task, registry) for task in spec.tasks
            },
            "strict_censoring_rule": "event_time < prediction_time",
            "exact_duplicate_fields": [
                "subject_id",
                "time",
                "code",
                "numeric_value",
                "text_value",
                "unit",
                "omop_table",
            ],
            "test_predictions_or_performance_used": False,
        },
        "cohort": cohort_summary,
        "cohort_sha256": cohort_hash,
        "event_data_sha256": actual_event_hash,
        "primary_model_repository": identity.repository,
        "primary_model_revision_hash": stable_hash(identity.revision),
        "primary_model_artifact_hash": identity.artifact_hash,
        "tokenizer_hash": identity.tokenizer_hash,
        "task_summaries": task_summaries,
        "outcome_and_shortcut_audits": outcome_audits,
        "count_baseline": {
            "status": count_report["status"],
            "report_sha256": file_sha256(count_report_path),
            "tasks": list(spec.tasks),
            "test_split_inspected": False,
        },
        "content_matching_passed": True,
        "label_index_only_perturbations_passed": True,
        "patient_window_metadata_alignment_passed": True,
        "pilot_artifacts_reused": False,
        "test_predictions_or_performance_used": False,
        "m2_run": False,
        "m3_run": False,
        "runtime_seconds": elapsed,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "private_storage_bytes": storage_bytes,
    }
    _write_immutable_json(spec.aggregate_root / "foundation.json", report, aggregate=True)
    return report
