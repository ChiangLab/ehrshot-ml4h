"""Protocol-aligned complete-history temporal-label conditions."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.blocks import PredictionWindow, clinical_signature

FULL_HISTORY_CONDITIONS = (
    "C0_order_only",
    "C1_correct_compact_time",
    "C2_reversed_earlier_gaps",
    "C3_shuffled_earlier_time_labels",
)


@dataclass(frozen=True)
class LabeledBlock:
    block: PredictionWindow
    label_window_index: int | None


@dataclass(frozen=True)
class TemporalCondition:
    name: str
    blocks: tuple[LabeledBlock, ...]
    eligible: bool
    perturbation_seed: int | None = None
    ineligibility_reason: str | None = None


def _seeded_random(seed: int, instance_key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{instance_key}|C3_shuffled_earlier_time_labels".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _labeled(
    name: str,
    blocks: Sequence[PredictionWindow],
    labels: Sequence[int | None],
    *,
    seed: int | None = None,
) -> TemporalCondition:
    return TemporalCondition(
        name,
        tuple(LabeledBlock(block, label) for block, label in zip(blocks, labels, strict=True)),
        True,
        seed,
    )


def _ineligible(
    name: str,
    blocks: Sequence[PredictionWindow],
    correct: Sequence[int],
    reason: str,
    *,
    seed: int | None = None,
) -> TemporalCondition:
    return TemporalCondition(
        name,
        tuple(LabeledBlock(block, label) for block, label in zip(blocks, correct, strict=True)),
        False,
        seed,
        reason,
    )


def build_full_history_conditions(
    blocks: Sequence[PredictionWindow],
    *,
    seed: int,
    instance_key: str,
    minimum_falsification_blocks: int = 3,
) -> dict[str, TemporalCondition]:
    """Build C0-C3 with identical content/order and seed-specific C3 labels."""
    blocks = tuple(blocks)
    correct = tuple(block.window_index for block in blocks)
    output = {
        "C0_order_only": _labeled("C0_order_only", blocks, (None,) * len(blocks)),
        "C1_correct_compact_time": _labeled("C1_correct_compact_time", blocks, correct),
    }
    short_reason = f"requires at least {minimum_falsification_blocks} observed prediction windows"
    if len(blocks) < minimum_falsification_blocks:
        output["C2_reversed_earlier_gaps"] = _ineligible(
            "C2_reversed_earlier_gaps", blocks, correct, short_reason
        )
        output["C3_shuffled_earlier_time_labels"] = _ineligible(
            "C3_shuffled_earlier_time_labels", blocks, correct, short_reason, seed=seed
        )
        assert_condition_invariants(output)
        return output

    gaps = tuple(correct[index] - correct[index + 1] for index in range(len(correct) - 1))
    falsified_gaps = tuple(reversed(gaps))
    reconstructed_newest_to_oldest = [correct[-1]]
    for gap in reversed(falsified_gaps):
        reconstructed_newest_to_oldest.append(reconstructed_newest_to_oldest[-1] + gap)
    reversed_labels = tuple(reversed(reconstructed_newest_to_oldest))
    if reversed_labels == correct:
        output["C2_reversed_earlier_gaps"] = _ineligible(
            "C2_reversed_earlier_gaps",
            blocks,
            correct,
            "reversing the observed gap sequence would not change an earlier label",
        )
    else:
        output["C2_reversed_earlier_gaps"] = _labeled(
            "C2_reversed_earlier_gaps", blocks, reversed_labels
        )

    shuffled_earlier = list(correct[:-1])
    rng = _seeded_random(seed, instance_key)
    rng.shuffle(shuffled_earlier)
    if shuffled_earlier == list(correct[:-1]):
        # A deterministic non-identity rotation preserves the seeded protocol while
        # avoiding an accidental no-op permutation for an otherwise eligible history.
        rotation = 1 + rng.randrange(len(shuffled_earlier) - 1)
        shuffled_earlier = shuffled_earlier[rotation:] + shuffled_earlier[:rotation]
    shuffled_labels = (*shuffled_earlier, correct[-1])
    if shuffled_labels == correct:
        output["C3_shuffled_earlier_time_labels"] = _ineligible(
            "C3_shuffled_earlier_time_labels",
            blocks,
            correct,
            "no changed earlier-label permutation exists",
            seed=seed,
        )
    else:
        output["C3_shuffled_earlier_time_labels"] = _labeled(
            "C3_shuffled_earlier_time_labels", blocks, shuffled_labels, seed=seed
        )
    assert_condition_invariants(output)
    return output


def _labels(condition: TemporalCondition) -> tuple[int | None, ...]:
    return tuple(item.label_window_index for item in condition.blocks)


def assert_condition_invariants(conditions: Mapping[str, TemporalCondition]) -> None:
    """Fail on content drift or a temporal perturbation outside its contract."""
    if tuple(conditions) != FULL_HISTORY_CONDITIONS:
        raise ContractError("The complete-history condition set or order is not canonical.")
    reference_blocks = tuple(item.block for item in conditions["C1_correct_compact_time"].blocks)
    reference_signature = clinical_signature(reference_blocks)
    correct = tuple(block.window_index for block in reference_blocks)
    if _labels(conditions["C1_correct_compact_time"]) != correct:
        raise ContractError("C1 labels do not match prediction-relative windows.")
    if _labels(conditions["C0_order_only"]) != (None,) * len(reference_blocks):
        raise ContractError("C0 must contain no elapsed-time labels.")
    for condition in conditions.values():
        if clinical_signature(tuple(item.block for item in condition.blocks)) != reference_signature:
            raise ContractError(f"{condition.name} changed clinical content or event order.")
    if len(reference_blocks) < 3:
        if any(conditions[name].eligible for name in FULL_HISTORY_CONDITIONS[2:]):
            raise ContractError("Too-short histories must be ineligible for falsified-time conditions.")
        return

    c2 = conditions["C2_reversed_earlier_gaps"]
    if c2.eligible:
        labels = tuple(int(value) for value in _labels(c2) if value is not None)
        correct_gaps = tuple(correct[index] - correct[index + 1] for index in range(len(correct) - 1))
        changed_gaps = tuple(labels[index] - labels[index + 1] for index in range(len(labels) - 1))
        if labels[-1] != correct[-1] or any(left <= right for left, right in pairwise(labels)):
            raise ContractError("C2 labels must be monotonic and retain the correct final label.")
        if sorted(changed_gaps) != sorted(correct_gaps):
            raise ContractError("C2 changed the observed inter-window gap multiset.")
        if labels[0] - labels[-1] != correct[0] - correct[-1]:
            raise ContractError("C2 changed the total observed history span.")
        if labels[:-1] == correct[:-1]:
            raise ContractError("Eligible C2 did not change an earlier label.")

    c3 = conditions["C3_shuffled_earlier_time_labels"]
    if c3.eligible:
        labels = tuple(int(value) for value in _labels(c3) if value is not None)
        if labels[-1] != correct[-1]:
            raise ContractError("C3 changed the final time label.")
        if sorted(labels[:-1]) != sorted(correct[:-1]):
            raise ContractError("C3 changed the earlier-label multiset.")
        if labels[:-1] == correct[:-1]:
            raise ContractError("Eligible C3 did not change an earlier label.")
        if any(label < min(correct) or label > max(correct) for label in labels):
            raise ContractError("C3 created a label outside the observed history span.")


def assert_shared_instance_metadata(metadata: Mapping[str, tuple[str, str, bool]]) -> None:
    """Assert that C0-C3 share task, private instance key, and outcome."""
    if tuple(metadata) != FULL_HISTORY_CONDITIONS:
        raise ContractError("Condition metadata set or order is not canonical.")
    if len(set(metadata.values())) != 1:
        raise ContractError("Task, patient instance, or outcome changed across temporal conditions.")
