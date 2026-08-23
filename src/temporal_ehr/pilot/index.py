"""Private patient-by-window index construction for the complete M1 pilot."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.history import HistoryInstance
from temporal_ehr.serialization.blocks import (
    PredictionWindow,
    build_prediction_windows,
    deduplicate_events,
    event_from_mapping,
    exact_duplicate_key,
    latest_code_occurrence,
    latest_window_only,
)
from temporal_ehr.serialization.conditions import LabeledBlock, TemporalCondition, build_full_history_conditions
from temporal_ehr.serialization.integrity import stable_hash
from temporal_ehr.serialization.packing import PackedHistory, PackingCache, content_matched_primary, pack_history


class InstructionTokenizer:
    """Count the frozen instruction and serialized history together."""

    def __init__(self, tokenizer: Any, instruction: str) -> None:
        self.tokenizer = tokenizer
        self.instruction = instruction.strip()

    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]:
        combined = f"{self.instruction}\n{text}" if text else self.instruction
        return self.tokenizer.encode(combined, add_special_tokens=add_special_tokens)

    def render(self, text: str) -> str:
        return f"{self.instruction}\n{text}" if text else self.instruction


@dataclass(frozen=True)
class PreparedInstance:
    index_rows: tuple[dict[str, Any], ...]
    text_rows: tuple[dict[str, Any], ...]
    source_event_count: int
    deduplicated_event_count: int


def _condition_from_windows(name: str, windows: Sequence[PredictionWindow]) -> TemporalCondition:
    return TemporalCondition(
        name=name,
        blocks=tuple(LabeledBlock(window, window.window_index) for window in windows),
        eligible=bool(windows),
        ineligibility_reason=None if windows else "no eligible source window",
    )


def _retained_by_window(history: PackedHistory) -> dict[int, int]:
    return {int(window_index): len(events) for window_index, events in history.retained_block_signature}


def _rows_for_representation(
    *,
    experiment_id: str,
    instance: HistoryInstance,
    source_windows: Sequence[PredictionWindow],
    representation: str,
    perturbation_seed: int | None,
    condition: TemporalCondition,
    packed: PackedHistory | None,
    rendered_text: str | None,
    window_days: int,
    model_hash: str,
    tokenizer_hash: str,
    instruction_hash: str,
    budget_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    displayed = {item.block.window_index: item.label_window_index for item in condition.blocks}
    retained = _retained_by_window(packed) if packed else {}
    text_hash = stable_hash(rendered_text) if rendered_text is not None else None
    representation_key = stable_hash((experiment_id, instance.instance_key, representation, perturbation_seed))
    index_rows: list[dict[str, Any]] = []
    width = dt.timedelta(days=window_days)
    for window in source_windows:
        retained_count = retained.get(window.window_index, 0)
        if not condition.eligible:
            retention_status = "INELIGIBLE"
        elif retained_count == 0:
            retention_status = "OMITTED"
        elif retained_count == len(window.events):
            retention_status = "RETAINED"
        else:
            retention_status = "PARTIAL"
        primary_key = stable_hash(
            (experiment_id, instance.instance_key, representation, perturbation_seed, window.window_index)
        )
        index_rows.append(
            {
                "index_row_key": primary_key,
                "experiment_id": experiment_id,
                "representation_key": representation_key,
                "instance_key": instance.instance_key,
                "subject_id": instance.subject_id,
                "task": instance.task,
                "split": instance.split,
                "prediction_time": instance.prediction_time,
                "target": instance.target,
                "window_days": window_days,
                "source_window_index": window.window_index,
                "window_start": instance.prediction_time - (window.window_index + 1) * width,
                "window_end": instance.prediction_time - window.window_index * width,
                "event_count": len(window.events),
                "retained_event_count": retained_count,
                "deduplicated_clinical_content_hash": stable_hash(
                    tuple(exact_duplicate_key(event) for event in window.events)
                ),
                "correct_time_label_index": window.window_index,
                "displayed_time_label_index": displayed.get(window.window_index),
                "representation": representation,
                "perturbation_seed": perturbation_seed,
                "retention_status": retention_status,
                "tokenizer_hash": tokenizer_hash,
                "model_hash": model_hash,
                "instruction_hash": instruction_hash,
                "budget_hash": budget_hash,
                "embedding_row_index": None,
                "cache_key": None,
            }
        )
    if not condition.eligible or packed is None or rendered_text is None:
        return index_rows, None
    source_times = [event.time for window in source_windows for event in window.events]
    retained_times = [signature[1] for signature in packed.retained_signature]
    return index_rows, {
        "representation_key": representation_key,
        "experiment_id": experiment_id,
        "instance_key": instance.instance_key,
        "subject_id": instance.subject_id,
        "task": instance.task,
        "split": instance.split,
        "prediction_time": instance.prediction_time,
        "target": instance.target,
        "representation": representation,
        "perturbation_seed": perturbation_seed,
        "eligible": True,
        "source_content_hash": stable_hash(
            tuple(exact_duplicate_key(event) for window in source_windows for event in window.events)
        ),
        "retained_content_hash": stable_hash(packed.retained_signature),
        "text_hash": text_hash,
        "text": rendered_text,
        "token_count": packed.token_count,
        "retained_event_count": packed.included_event_count,
        "source_event_count": packed.total_event_count,
        "source_historical_span_days": (
            instance.prediction_time - min(source_times)
        ).total_seconds()
        / 86_400,
        "retained_historical_span_days": (
            (instance.prediction_time - min(retained_times)).total_seconds() / 86_400
            if retained_times
            else 0.0
        ),
        "truncated": packed.truncated,
        "embedding_row_index": None,
        "cache_key": None,
    }


def prepare_instance(
    instance: HistoryInstance,
    *,
    experiment_id: str,
    window_days: int,
    c3_seeds: Sequence[int],
    descriptions: Mapping[str, str],
    tokenizer: Any,
    instruction: str,
    budget: int,
    model_hash: str,
    tokenizer_hash: str,
) -> PreparedInstance:
    """Create private representation and window-index rows without model inference."""
    if instance.split not in {"train", "validation"}:
        raise ContractError("Pilot preparation prohibits the test split.")
    events = tuple(event_from_mapping(row) for row in instance.events)
    if any(event.time >= instance.prediction_time for event in events):
        raise ContractError("Pilot preparation received a future or exact-prediction-time event.")
    deduplicated, _ = deduplicate_events(events)
    windows = build_prediction_windows(deduplicated, instance.prediction_time, window_days=window_days)
    if not windows:
        raise ContractError("Pilot instance has no strictly prior event window.")
    prefixed = InstructionTokenizer(tokenizer, instruction)
    instruction_hash = stable_hash(instruction.strip())
    budget_hash = stable_hash({"token_budget": budget})
    seed_conditions = {
        int(seed): build_full_history_conditions(windows, seed=int(seed), instance_key=instance.instance_key)
        for seed in c3_seeds
    }
    first = seed_conditions[int(c3_seeds[0])]
    variants: list[tuple[str, int | None, TemporalCondition]] = [
        ("C0_order_only", None, first["C0_order_only"]),
        ("C1_correct_compact_time", None, first["C1_correct_compact_time"]),
        ("C2_reversed_earlier_gaps", None, first["C2_reversed_earlier_gaps"]),
        *[
            ("C3_shuffled_earlier_time_labels", int(seed), conditions["C3_shuffled_earlier_time_labels"])
            for seed, conditions in seed_conditions.items()
        ],
    ]
    aliases = {
        f"{name}::{seed if seed is not None else 'none'}": condition
        for name, seed, condition in variants
        if condition.eligible
    }
    shared = content_matched_primary(
        aliases,
        descriptions,
        tokenizer=prefixed,
        budget=budget,
        cache=PackingCache(descriptions, prefixed),
    )
    baseline_variants = (
        ("B1_latest_window_only", _condition_from_windows("B1_latest_window_only", latest_window_only(windows))),
        (
            "B2_latest_code_occurrence",
            _condition_from_windows("B2_latest_code_occurrence", latest_code_occurrence(windows)),
        ),
    )
    index_rows: list[dict[str, Any]] = []
    text_rows: list[dict[str, Any]] = []
    for name, seed, condition in variants:
        alias = f"{name}::{seed if seed is not None else 'none'}"
        packed = shared.histories.get(alias)
        rendered = prefixed.render(packed.text) if packed else None
        rows, text_row = _rows_for_representation(
            experiment_id=experiment_id,
            instance=instance,
            source_windows=windows,
            representation=name,
            perturbation_seed=seed,
            condition=condition,
            packed=packed,
            rendered_text=rendered,
            window_days=window_days,
            model_hash=model_hash,
            tokenizer_hash=tokenizer_hash,
            instruction_hash=instruction_hash,
            budget_hash=budget_hash,
        )
        index_rows.extend(rows)
        if text_row:
            text_rows.append(text_row)
    for name, condition in baseline_variants:
        packed = pack_history(condition, descriptions, tokenizer=prefixed, budget=budget)
        rendered = prefixed.render(packed.text)
        rows, text_row = _rows_for_representation(
            experiment_id=experiment_id,
            instance=instance,
            source_windows=windows,
            representation=name,
            perturbation_seed=None,
            condition=condition,
            packed=packed,
            rendered_text=rendered,
            window_days=window_days,
            model_hash=model_hash,
            tokenizer_hash=tokenizer_hash,
            instruction_hash=instruction_hash,
            budget_hash=budget_hash,
        )
        index_rows.extend(rows)
        assert text_row is not None
        text_rows.append(text_row)
    return PreparedInstance(tuple(index_rows), tuple(text_rows), len(events), len(deduplicated))


def finalize_indices(
    index_rows: Sequence[dict[str, Any]],
    text_rows: Sequence[dict[str, Any]],
    *,
    experiment_id: str,
    model_hash: str,
    tokenizer_hash: str,
    instructions: Mapping[str, str],
    budget: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assign deterministic embedding rows/cache keys and assert one-to-one alignment."""
    texts = sorted(
        (dict(row) for row in text_rows),
        key=lambda row: (
            row["task"],
            row["split"],
            row["instance_key"],
            row["representation"],
            row["perturbation_seed"] if row["perturbation_seed"] is not None else -1,
        ),
    )
    representation_lookup: dict[str, tuple[int, str]] = {}
    for row_index, row in enumerate(texts):
        cache_key = stable_hash(
            {
                "experiment_id": experiment_id,
                "task": row["task"],
                "instance_key": row["instance_key"],
                "representation": row["representation"],
                "perturbation_seed": row["perturbation_seed"],
                "text_hash": row["text_hash"],
                "model_hash": model_hash,
                "tokenizer_hash": tokenizer_hash,
                "instruction_hash": stable_hash(instructions[row["task"]].strip()),
                "budget": budget,
            }
        )
        row["embedding_row_index"] = row_index
        row["cache_key"] = cache_key
        if row["representation_key"] in representation_lookup:
            raise ContractError("Duplicate prepared representation key.")
        representation_lookup[row["representation_key"]] = (row_index, cache_key)
    indices = sorted(
        (dict(row) for row in index_rows),
        key=lambda row: (
            row["task"],
            row["split"],
            row["instance_key"],
            row["representation"],
            row["perturbation_seed"] if row["perturbation_seed"] is not None else -1,
            -row["source_window_index"],
        ),
    )
    primary_keys: set[str] = set()
    for row in indices:
        if row["index_row_key"] in primary_keys:
            raise ContractError("Patient-window index primary key is not unique.")
        primary_keys.add(row["index_row_key"])
        aligned = representation_lookup.get(row["representation_key"])
        if row["retention_status"] == "INELIGIBLE":
            if aligned is not None:
                raise ContractError("Ineligible representation unexpectedly has an embedding row.")
            continue
        if aligned is None:
            raise ContractError("Eligible patient-window row lacks a prepared text/embedding row.")
        row["embedding_row_index"], row["cache_key"] = aligned
    return indices, texts
