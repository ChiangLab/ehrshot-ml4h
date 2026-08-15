"""Aggregate-only serialization quality control over governed histories."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from temporal_ehr.data.contracts import assert_privacy_safe_report
from temporal_ehr.data.history import MODELING_SPLITS, iter_strict_histories
from temporal_ehr.serialization.blocks import build_calendar_blocks, event_from_mapping, latest_window_only
from temporal_ehr.serialization.conditions import (
    FULL_HISTORY_CONDITIONS,
    assert_condition_invariants,
    assert_shared_instance_metadata,
    build_full_history_conditions,
)
from temporal_ehr.serialization.packing import Tokenizer, pack_history
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


def _summarize_groups(
    groups: dict[tuple[Any, ...], dict[str, list[int | float]]],
    *,
    minimum_group_size: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for key in sorted(groups):
        values = groups[key]
        group_size = len(values["event_count"])
        identity = {"task": key[0], "split": key[1], "window_days": key[2]}
        if group_size < minimum_group_size:
            summaries.append(
                {
                    **identity,
                    "status": "SUPPRESSED",
                    "reason": f"group size is below minimum {minimum_group_size}",
                }
            )
            continue
        summaries.append(
            {
                **identity,
                "status": "REPORTED",
                "instances": group_size,
                **{name: _distribution(metric_values) for name, metric_values in values.items()},
            }
        )
    return summaries


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
    """Compute only privacy-safe distributions; real serialized text is never retained."""
    if tokenizer is None:
        raise RuntimeError(
            "Real-data serialization QC requires the already-downloaded Qwen tokenizer. "
            "A whitespace estimate is permitted only for synthetic demonstrations."
        )
    descriptions = load_code_descriptions(asset_root)
    window_groups: dict[tuple[Any, ...], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))
    condition_groups: dict[tuple[str, str, str], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))
    latest_groups: dict[tuple[str, str, int], dict[str, list[int | float]]] = defaultdict(lambda: defaultdict(list))

    for instance in iter_strict_histories(event_path, cohort_path, tasks=tasks, splits=splits):
        events = tuple(event_from_mapping(row) for row in instance.events)
        day_blocks = build_calendar_blocks(events, instance.prediction_time, window_days=1)
        history_span = day_blocks[0].relative_day - day_blocks[-1].relative_day if day_blocks else 0
        for window_days in (1, 30, 90):
            blocks = build_calendar_blocks(events, instance.prediction_time, window_days=window_days)
            group = window_groups[(instance.task, instance.split, window_days)]
            group["event_count"].append(len(events))
            group["block_count"].append(len(blocks))
            group["history_span_days"].append(history_span)

        conditions = build_full_history_conditions(
            day_blocks,
            seed=perturbation_seed,
            instance_key=instance.instance_key,
        )
        assert_condition_invariants(conditions)
        assert_shared_instance_metadata(
            {name: (instance.instance_key, instance.target) for name in FULL_HISTORY_CONDITIONS}
        )
        packed_by_name = {}
        for name in FULL_HISTORY_CONDITIONS:
            condition = conditions[name]
            group = condition_groups[(instance.task, instance.split, name)]
            group["event_count"].append(len(events))
            group["block_count"].append(len(day_blocks))
            group["history_span_days"].append(history_span)
            group["eligible"].append(int(condition.eligible))
            if condition.eligible:
                packed = pack_history(condition, descriptions, tokenizer=tokenizer, budget=token_budget)
                packed_by_name[name] = packed
                group["token_count"].append(packed.token_count)
                group["truncation_rate"].append(int(packed.truncated))
        if "order_only" in packed_by_name and "correct_compact_time" in packed_by_name:
            displaced = max(
                0,
                packed_by_name["order_only"].included_event_count
                - packed_by_name["correct_compact_time"].included_event_count,
            )
            condition_groups[(instance.task, instance.split, "correct_compact_time")][
                "clinical_events_displaced_by_time_tokens"
            ].append(displaced)

        for recent_days in (30, 90):
            reduced = latest_window_only(day_blocks, window_days=recent_days)
            retained = sum(len(block.events) for block in reduced)
            group = latest_groups[(instance.task, instance.split, recent_days)]
            group["event_count"].append(len(events))
            group["retained_event_count"].append(retained)
            group["retained_block_count"].append(len(reduced))

    window_summary = _summarize_groups(window_groups, minimum_group_size=minimum_group_size)
    latest_summary = _summarize_groups(latest_groups, minimum_group_size=minimum_group_size)
    condition_summary: list[dict[str, Any]] = []
    for key in sorted(condition_groups):
        values = condition_groups[key]
        group_size = len(values["event_count"])
        identity = {"task": key[0], "split": key[1], "condition": key[2]}
        if group_size < minimum_group_size:
            condition_summary.append(
                {
                    **identity,
                    "status": "SUPPRESSED",
                    "reason": f"group size is below minimum {minimum_group_size}",
                }
            )
        else:
            condition_summary.append(
                {
                    **identity,
                    "status": "REPORTED",
                    "instances": group_size,
                    **{name: _distribution(metric_values) for name, metric_values in values.items()},
                }
            )

    report: dict[str, Any] = {
        "status": "PASS",
        "data_scope": "real EHRSHOT aggregate QC; train and validation only",
        "strict_censoring_rule": "event_time < prediction_time",
        "atomic_block": "calendar day",
        "aggregation_windows_are_qc_only": [30, 90],
        "primary_window_selected": False,
        "token_budget": token_budget,
        "token_measurement": "exact counts from the configured local Qwen tokenizer",
        "full_history_conditions": list(FULL_HISTORY_CONDITIONS),
        "patient_outcome_alignment_asserted": True,
        "condition_distributions": condition_summary,
        "full_history_block_window_comparison": window_summary,
        "latest_window_only_information_reduction": {
            "is_full_history_condition": False,
            "distributions": latest_summary,
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "serialization_qc.json").write_text(text, encoding="utf-8")
    return report
