"""Controlled full-history temporal-label conditions."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.blocks import CalendarBlock, clinical_signature

FULL_HISTORY_CONDITIONS = (
    "order_only",
    "correct_compact_time",
    "reversed_earlier_gaps",
    "shuffled_earlier_time_labels",
)


@dataclass(frozen=True)
class LabeledBlock:
    block: CalendarBlock
    time_label_days: int | None


@dataclass(frozen=True)
class TemporalCondition:
    name: str
    blocks: tuple[LabeledBlock, ...]
    eligible: bool
    ineligibility_reason: str | None = None


def _seeded_random(seed: int, instance_key: str, condition: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{instance_key}|{condition}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _labeled(name: str, blocks: Sequence[CalendarBlock], labels: Sequence[int | None]) -> TemporalCondition:
    return TemporalCondition(
        name, tuple(LabeledBlock(block, label) for block, label in zip(blocks, labels, strict=True)), True
    )


def build_full_history_conditions(
    blocks: Sequence[CalendarBlock],
    *,
    seed: int,
    instance_key: str,
    minimum_falsification_blocks: int = 3,
) -> dict[str, TemporalCondition]:
    """Build four conditions while holding clinical content and order fixed."""
    blocks = tuple(blocks)
    correct = tuple(block.relative_day for block in blocks)
    output = {
        "order_only": _labeled("order_only", blocks, (None,) * len(blocks)),
        "correct_compact_time": _labeled("correct_compact_time", blocks, correct),
    }
    if len(blocks) < minimum_falsification_blocks:
        reason = f"requires at least {minimum_falsification_blocks} observed blocks"
        for name in ("reversed_earlier_gaps", "shuffled_earlier_time_labels"):
            output[name] = TemporalCondition(
                name,
                tuple(LabeledBlock(block, label) for block, label in zip(blocks, correct, strict=True)),
                False,
                reason,
            )
        return output

    gaps = [correct[index] - correct[index + 1] for index in range(len(correct) - 1)]
    reversed_labels = [correct[-1]]
    for gap in gaps:
        reversed_labels.append(reversed_labels[-1] + gap)
    reversed_labels.reverse()

    shuffled_earlier = list(correct[:-1])
    rng = _seeded_random(seed, instance_key, "shuffled_earlier_time_labels")
    rng.shuffle(shuffled_earlier)
    if shuffled_earlier == list(correct[:-1]) and len(shuffled_earlier) > 1:
        shuffled_earlier = shuffled_earlier[1:] + shuffled_earlier[:1]
    shuffled_labels = (*shuffled_earlier, correct[-1])

    output["reversed_earlier_gaps"] = _labeled("reversed_earlier_gaps", blocks, reversed_labels)
    if tuple(reversed_labels) == correct:
        output["reversed_earlier_gaps"] = TemporalCondition(
            "reversed_earlier_gaps",
            output["reversed_earlier_gaps"].blocks,
            False,
            "reversing symmetric observed gaps would not change the label mapping",
        )
    output["shuffled_earlier_time_labels"] = _labeled("shuffled_earlier_time_labels", blocks, shuffled_labels)
    assert_condition_invariants(output)
    return output


def assert_condition_invariants(conditions: Mapping[str, TemporalCondition]) -> None:
    """Fail if a temporal perturbation changes content/order or violates its label contract."""
    if tuple(conditions) != FULL_HISTORY_CONDITIONS:
        raise ContractError("The full-history condition set or order is not canonical.")
    reference_blocks = tuple(item.block for item in conditions["correct_compact_time"].blocks)
    reference_signature = clinical_signature(reference_blocks)
    correct_labels = tuple(item.time_label_days for item in conditions["correct_compact_time"].blocks)
    expected_labels = tuple(block.relative_day for block in reference_blocks)
    if correct_labels != expected_labels:
        raise ContractError("correct_compact_time labels do not match the prediction-anchored blocks.")
    if tuple(item.time_label_days for item in conditions["order_only"].blocks) != (None,) * len(reference_blocks):
        raise ContractError("order_only must contain no elapsed-time labels.")
    for condition in conditions.values():
        if clinical_signature(tuple(item.block for item in condition.blocks)) != reference_signature:
            raise ContractError(f"{condition.name} changed clinical content or event order.")
    if len(reference_blocks) < 3:
        if any(conditions[name].eligible for name in FULL_HISTORY_CONDITIONS[2:]):
            raise ContractError("Too-short histories must be ineligible for falsified-time conditions.")
        return
    final_label = correct_labels[-1]
    lower = min(label for label in correct_labels if label is not None)
    upper = max(label for label in correct_labels if label is not None)
    for name in FULL_HISTORY_CONDITIONS[2:]:
        labels = tuple(item.time_label_days for item in conditions[name].blocks)
        if labels[-1] != final_label:
            raise ContractError(f"{name} changed the final time label.")
        if any(label is None or not lower <= label <= upper for label in labels):
            raise ContractError(f"{name} created a label outside the observed history span.")


def assert_shared_instance_metadata(metadata: Mapping[str, tuple[str, bool]]) -> None:
    """Assert that all four conditions refer to the same cohort instance and outcome."""
    if tuple(metadata) != FULL_HISTORY_CONDITIONS:
        raise ContractError("Condition metadata set or order is not canonical.")
    if len(set(metadata.values())) != 1:
        raise ContractError("Patient instance or outcome changed across temporal conditions.")
