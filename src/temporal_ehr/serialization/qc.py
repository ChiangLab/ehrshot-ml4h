"""Privacy-safe aggregate protocol review over governed train/validation histories."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from temporal_ehr.data.contracts import assert_privacy_safe_report
from temporal_ehr.data.history import MODELING_SPLITS, iter_strict_histories
from temporal_ehr.data.inventory import sha256_file
from temporal_ehr.experiments import load_experiment_panel, window_days_for_task
from temporal_ehr.serialization.blocks import (
    build_prediction_windows,
    deduplicate_events,
    event_from_mapping,
    latest_code_occurrence,
    latest_window_only,
)
from temporal_ehr.serialization.conditions import (
    FULL_HISTORY_CONDITIONS,
    assert_condition_invariants,
    assert_shared_instance_metadata,
    build_full_history_conditions,
)
from temporal_ehr.serialization.integrity import (
    build_instance_manifest,
    require_aggregate_output,
    require_private_output,
    stable_hash,
)
from temporal_ehr.serialization.packing import (
    Tokenizer,
    content_matched_primary,
    independent_operational,
)
from temporal_ehr.serialization.render import load_code_descriptions


def _distribution(values: Sequence[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"minimum": None, "p25": None, "median": None, "mean": None, "p75": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _summaries(
    groups: Mapping[tuple[Any, ...], Mapping[str, list[int | float]]],
    identity_fields: Sequence[str],
    *,
    minimum_group_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda item: tuple(str(value) for value in item)):
        values = groups[key]
        group_size = len(next(iter(values.values()))) if values else 0
        identity = dict(zip(identity_fields, key, strict=True))
        if group_size < minimum_group_size:
            output.append(
                {
                    **identity,
                    "status": "SUPPRESSED",
                    "reason": f"group size is below minimum {minimum_group_size}",
                }
            )
        else:
            output.append(
                {
                    **identity,
                    "status": "REPORTED",
                    "instances": group_size,
                    **{name: _distribution(metric) for name, metric in values.items()},
                }
            )
    return output


def _read_revision(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "REVISION_FILE_NOT_PRESENT"


def run_protocol_alignment_qc(
    event_path: Path,
    cohort_path: Path,
    asset_root: Path,
    *,
    tokenizer: Tokenizer,
    tokenizer_root: Path,
    registry: Mapping[str, Any],
    tasks: Sequence[str],
    splits: Sequence[str] = MODELING_SPLITS,
    token_budgets: Sequence[int] = (4_096, 8_192),
    c3_seeds: Sequence[int] = (20260815, 20260816, 20260817),
    minimum_group_size: int = 20,
    report_path: Path = Path("reports/generated/phase1_protocol_review/protocol_alignment.json"),
    private_manifest_path: Path = Path("outputs/private/manifests/phase1_protocol_alignment.jsonl"),
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run exact-time serializer QC without embeddings or test-set modeling."""
    if tokenizer is None:
        raise RuntimeError("Protocol review requires an already-downloaded local Qwen tokenizer.")
    if tuple(splits) != MODELING_SPLITS:
        raise ValueError("Protocol review is restricted to train and validation in canonical order.")
    if tuple(token_budgets) != (4_096, 8_192):
        raise ValueError("Protocol review requires the prespecified 4096 and 8192 token budgets.")
    if tuple(c3_seeds) != tuple(registry["serialization"]["c3_perturbation_seeds"]):
        raise ValueError("C3 seeds must match configs/tasks.yaml exactly.")

    root = (repository_root or Path.cwd()).resolve()
    destination = require_aggregate_output(report_path, root)
    manifest_path = require_private_output(private_manifest_path, root)
    descriptions = load_code_descriptions(asset_root)
    revision_path = tokenizer_root / "PINNED_REVISION.txt"
    tokenizer_revision = _read_revision(revision_path)
    dataset_revision = sha256_file(event_path)
    config_revision = stable_hash(registry)

    duplicate_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))
    window_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))
    eligibility_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ineligibility_reasons: Counter[tuple[str, str, str, int, str]] = Counter()
    matched_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))
    independent_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    baseline_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content_match_failures = 0
    instance_count = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        for instance in iter_strict_histories(event_path, cohort_path, tasks=tasks, splits=splits):
            instance_count += 1
            source_events = tuple(event_from_mapping(row) for row in instance.events)
            events, duplicate_count = deduplicate_events(source_events)
            duplicate_group = duplicate_groups[(instance.task, instance.split)]
            duplicate_group["source_event_count"].append(len(source_events))
            duplicate_group["exact_duplicates_removed"].append(duplicate_count)
            duplicate_group["post_deduplication_event_count"].append(len(events))

            primary_window_days = window_days_for_task(instance.task, registry)
            primary_windows = build_prediction_windows(
                events, instance.prediction_time, window_days=primary_window_days
            )
            for width in (30, 90):
                windows = build_prediction_windows(events, instance.prediction_time, window_days=width)
                group = window_groups[(instance.task, instance.split, width)]
                group["event_count"].append(len(events))
                group["nonempty_window_count"].append(len(windows))
                group["history_span_window_count"].append(
                    windows[0].window_index - windows[-1].window_index + 1 if windows else 0
                )

            latest_window = latest_window_only(primary_windows)
            latest_code = latest_code_occurrence(primary_windows)
            for baseline_name, retained in (
                ("latest_window_only", latest_window),
                ("latest_code_occurrence", latest_code),
            ):
                retained_count = sum(len(block.events) for block in retained)
                group = baseline_groups[(instance.task, instance.split, baseline_name)]
                group["source_event_count"].append(len(events))
                group["retained_event_count"].append(retained_count)
                group["retained_fraction"].append(retained_count / len(events) if events else 0.0)
                if retained:
                    baseline_conditions = build_full_history_conditions(
                        retained, seed=int(c3_seeds[0]), instance_key=instance.instance_key
                    )
                    final = baseline_conditions["C1_correct_compact_time"].blocks[-1]
                    if final.label_window_index != retained[-1].window_index:
                        raise RuntimeError("Information-reduction baseline lost the correct final-window label.")

            for seed in c3_seeds:
                conditions = build_full_history_conditions(
                    primary_windows,
                    seed=int(seed),
                    instance_key=instance.instance_key,
                )
                assert_condition_invariants(conditions)
                assert_shared_instance_metadata(
                    {
                        name: (instance.task, instance.instance_key, instance.target)
                        for name in FULL_HISTORY_CONDITIONS
                    }
                )
                for name in FULL_HISTORY_CONDITIONS[2:]:
                    group = eligibility_groups[(instance.task, instance.split, name, int(seed))]
                    group["eligible"].append(int(conditions[name].eligible))
                    if not conditions[name].eligible:
                        ineligibility_reasons[
                            (
                                instance.task,
                                instance.split,
                                name,
                                int(seed),
                                str(conditions[name].ineligibility_reason),
                            )
                        ] += 1

                for budget in token_budgets:
                    matched = content_matched_primary(
                        conditions, descriptions, tokenizer=tokenizer, budget=int(budget)
                    )
                    signatures = {history.retained_signature for history in matched.histories.values()}
                    block_signatures = {
                        history.retained_block_signature for history in matched.histories.values()
                    }
                    if len(signatures) != 1 or len(block_signatures) != 1:
                        content_match_failures += 1
                    operational = independent_operational(
                        conditions, descriptions, tokenizer=tokenizer, budget=int(budget)
                    )
                    manifests: list[dict[str, Any]] = []
                    matched_c0_tokens = matched.histories["C0_order_only"].token_count
                    for name, history in matched.histories.items():
                        group = matched_groups[(instance.task, instance.split, name, int(seed), int(budget))]
                        group["content_match_success"].append(int(len(signatures) == 1))
                        group["retained_event_count"].append(history.included_event_count)
                        group["retained_fraction"].append(
                            history.included_event_count / history.total_event_count
                            if history.total_event_count
                            else 0.0
                        )
                        group["token_count"].append(history.token_count)
                        group["time_label_token_cost_vs_C0"].append(history.token_count - matched_c0_tokens)
                        manifest = build_instance_manifest(
                            dataset_revision=dataset_revision,
                            config_revision=config_revision,
                            model_revision="NO_EMBEDDING_MODEL_USED_IN_PHASE2",
                            tokenizer_revision=tokenizer_revision,
                            task=instance.task,
                            split=instance.split,
                            outcome=instance.target,
                            subject_id=instance.subject_id,
                            instance_key=instance.instance_key,
                            prediction_time=instance.prediction_time,
                            source_events=source_events,
                            deduplicated_events=events,
                            retained_signature=history.retained_signature,
                            retained_block_signature=history.retained_block_signature,
                            rendered_text=history.text,
                            condition=name,
                            perturbation_seed=conditions[name].perturbation_seed,
                            comparison_seed=int(seed),
                            token_budget=int(budget),
                        )
                        manifests.append(manifest)
                    identity_fields = (
                        "task",
                        "canonical_split",
                        "patient_hash",
                        "prediction_instance_hash",
                        "outcome_hash",
                        "retained_exact_event_hashes",
                        "retained_block_content_hash",
                    )
                    identities = {
                        tuple(json.dumps(manifest[field], sort_keys=True) for field in identity_fields)
                        for manifest in manifests
                    }
                    if len(identities) != 1:
                        raise RuntimeError("Content-matched manifest identity or content drift was detected.")
                    for manifest in manifests:
                        manifest_handle.write(json.dumps(manifest, sort_keys=True) + "\n")
                    operational_c0_tokens = operational.histories["C0_order_only"].token_count
                    for name, history in operational.histories.items():
                        group = independent_groups[(instance.task, instance.split, name, int(seed), int(budget))]
                        group["retained_event_count"].append(history.included_event_count)
                        group["token_count"].append(history.token_count)
                        group["events_displaced_vs_C0"].append(operational.displaced_events[name])
                        group["token_count_delta_vs_C0"].append(history.token_count - operational_c0_tokens)

    panel_paths = {
        "pilot_three_task": root / "configs/experiments/pilot_three_task.yaml",
        "proposed_four_task": root / "configs/experiments/proposed_four_task.yaml",
        "locked_panel": root / "configs/experiments/locked_panel.yaml",
    }
    panel_status = {name: load_experiment_panel(path)["status"] for name, path in panel_paths.items()}
    checklist = {
        "train_validation_only": True,
        "test_predictions_or_performance_used": False,
        "exact_duplicates_removed_before_serialization": True,
        "prediction_relative_windows_configured": True,
        "three_c3_seeds_checked": len(tuple(c3_seeds)) == 3,
        "content_matching_passed": content_match_failures == 0,
        "private_integrity_manifests_written": manifest_path.is_file(),
        "experiment_panel_locked": panel_status["locked_panel"] == "LOCKED",
        "embedding_experiment_run": False,
    }
    checklist["phase3_go"] = all(
        checklist[name]
        for name in (
            "train_validation_only",
            "exact_duplicates_removed_before_serialization",
            "prediction_relative_windows_configured",
            "three_c3_seeds_checked",
            "content_matching_passed",
            "private_integrity_manifests_written",
            "experiment_panel_locked",
        )
    )
    report: dict[str, Any] = {
        "status": "PASS" if content_match_failures == 0 else "FAIL",
        "protocol_decision": "NO_GO" if not checklist["phase3_go"] else "GO",
        "data_scope": "real EHRSHOT aggregate QC; train and validation only",
        "strict_censoring_rule": "event_time < prediction_time",
        "window_intervals": "[T-(i+1)w, T-iw); T is prediction time and is excluded",
        "family_window_days": dict(registry["serialization"]["family_window_days"]),
        "exact_duplicate_key": [
            "subject_id",
            "time",
            "code",
            "numeric_value",
            "text_value",
            "unit",
            "omop_table",
        ],
        "panel_status": panel_status,
        "tasks_reviewed": list(tasks),
        "instances_reviewed": instance_count,
        "c3_seeds": [int(seed) for seed in c3_seeds],
        "token_budgets": [int(budget) for budget in token_budgets],
        "token_measurement": "exact counts from the configured local Qwen tokenizer",
        "duplicate_removal": _summaries(
            duplicate_groups, ("task", "split"), minimum_group_size=minimum_group_size
        ),
        "prediction_window_distributions": _summaries(
            window_groups, ("task", "split", "window_days"), minimum_group_size=minimum_group_size
        ),
        "condition_eligibility": _summaries(
            eligibility_groups,
            ("task", "split", "condition", "c3_seed"),
            minimum_group_size=minimum_group_size,
        ),
        "ineligibility_reason_counts": [
            {
                "task": task,
                "split": split,
                "condition": condition,
                "c3_seed": seed,
                "reason": reason,
                "count": count,
            }
            for (task, split, condition, seed, reason), count in sorted(ineligibility_reasons.items())
            if count >= minimum_group_size
        ],
        "content_matched_primary": _summaries(
            matched_groups,
            ("task", "split", "condition", "c3_seed", "token_budget"),
            minimum_group_size=minimum_group_size,
        ),
        "independent_operational": _summaries(
            independent_groups,
            ("task", "split", "condition", "c3_seed", "token_budget"),
            minimum_group_size=minimum_group_size,
        ),
        "information_reduction_baselines": _summaries(
            baseline_groups,
            ("task", "split", "baseline"),
            minimum_group_size=minimum_group_size,
        ),
        "private_manifest_logical_path": "outputs/private/manifests/phase1_protocol_alignment.jsonl",
        "checklist": checklist,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return report


def run_serialization_qc(
    event_path: Path,
    cohort_path: Path,
    asset_root: Path,
    *,
    tokenizer: Tokenizer | None,
    tasks: Sequence[str],
    splits: Sequence[str] = MODELING_SPLITS,
    token_budget: int = 4_096,
    perturbation_seed: int = 20260815,
    minimum_group_size: int = 20,
    report_root: Path = Path("reports/generated/serialization_qc"),
) -> dict[str, Any]:
    """Compatibility wrapper for Notebook 02 using the complete Phase 2 audit."""
    del token_budget, perturbation_seed, report_root
    if tokenizer is None:
        raise RuntimeError("Real-data serialization QC requires the already-downloaded Qwen tokenizer.")
    from temporal_ehr.data.task_registry import load_task_registry

    registry = load_task_registry(Path("configs/tasks.yaml"))
    tokenizer_root = Path(str(getattr(tokenizer, "name_or_path", ".")))
    return run_protocol_alignment_qc(
        event_path,
        cohort_path,
        asset_root,
        tokenizer=tokenizer,
        tokenizer_root=tokenizer_root,
        registry=registry,
        tasks=tasks,
        splits=splits,
        minimum_group_size=minimum_group_size,
    )
