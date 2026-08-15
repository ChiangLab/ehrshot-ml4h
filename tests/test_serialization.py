from __future__ import annotations

import datetime as dt

import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.blocks import (
    ClinicalEvent,
    build_calendar_blocks,
    clinical_signature,
    latest_window_only,
)
from temporal_ehr.serialization.conditions import (
    FULL_HISTORY_CONDITIONS,
    assert_condition_invariants,
    assert_shared_instance_metadata,
    build_full_history_conditions,
)
from temporal_ehr.serialization.packing import pack_history, whitespace_estimate_for_synthetic_demo
from temporal_ehr.serialization.render import describe_code, render_condition


class WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def _events(prediction: dt.datetime) -> tuple[ClinicalEvent, ...]:
    return (
        ClinicalEvent(prediction - dt.timedelta(days=100), "SYN/A", omop_table="measurement"),
        ClinicalEvent(prediction - dt.timedelta(days=60, hours=2), "SYN/R", numeric_value=1.0),
        ClinicalEvent(prediction - dt.timedelta(days=60, hours=1), "SYN/R", numeric_value=2.0),
        ClinicalEvent(prediction - dt.timedelta(days=10), "SYN/Z"),
    )


def test_calendar_blocks_preserve_recurrence_and_deterministic_concurrency() -> None:
    prediction = dt.datetime(2024, 1, 1, 12)
    concurrent = (
        ClinicalEvent(prediction - dt.timedelta(days=10), "SYN/Z"),
        ClinicalEvent(prediction - dt.timedelta(days=10), "SYN/A"),
    )
    blocks = build_calendar_blocks((*_events(prediction), *concurrent), prediction)
    assert [block.relative_day for block in blocks] == [100, 60, 10]
    assert sum(event.code == "SYN/R" for event in blocks[1].events) == 2
    assert [event.code for event in blocks[-1].events] == ["SYN/A", "SYN/Z", "SYN/Z"]


def test_block_construction_rejects_exact_prediction_time() -> None:
    prediction = dt.datetime(2024, 1, 1)
    with pytest.raises(ContractError, match="at or after"):
        build_calendar_blocks((ClinicalEvent(prediction, "SYN/EXACT"),), prediction)


@pytest.mark.parametrize("block_count", [1, 2])
def test_short_histories_are_marked_ineligible(block_count: int) -> None:
    prediction = dt.datetime(2024, 1, 1)
    blocks = build_calendar_blocks(_events(prediction)[:block_count], prediction)
    conditions = build_full_history_conditions(blocks, seed=4, instance_key="invented")
    assert not conditions["reversed_earlier_gaps"].eligible
    assert not conditions["shuffled_earlier_time_labels"].eligible
    assert_condition_invariants(conditions)


def test_temporal_conditions_are_deterministic_and_preserve_every_clinical_field() -> None:
    prediction = dt.datetime(2024, 1, 1, 12)
    blocks = build_calendar_blocks(_events(prediction), prediction)
    first = build_full_history_conditions(blocks, seed=42, instance_key="invented")
    second = build_full_history_conditions(blocks, seed=42, instance_key="invented")
    assert first == second
    assert_condition_invariants(first)
    signature = clinical_signature(blocks)
    assert all(
        clinical_signature(tuple(item.block for item in condition.blocks)) == signature for condition in first.values()
    )
    assert all(item.time_label_days is None for item in first["order_only"].blocks)
    assert all(tuple(item.block for item in condition.blocks) == blocks for condition in first.values())
    correct = [item.time_label_days for item in first["correct_compact_time"].blocks]
    reversed_labels = [item.time_label_days for item in first["reversed_earlier_gaps"].blocks]
    shuffled = [item.time_label_days for item in first["shuffled_earlier_time_labels"].blocks]
    assert correct == [100, 60, 10]
    assert reversed_labels == [100, 50, 10]
    assert shuffled[-1] == correct[-1]
    assert sorted(shuffled[:-1]) == sorted(correct[:-1])
    assert all(min(correct) <= label <= max(correct) for label in (*reversed_labels, *shuffled))


def test_condition_patient_and_outcome_alignment_is_asserted_directly() -> None:
    aligned = {name: ("invented-instance", True) for name in FULL_HISTORY_CONDITIONS}
    assert_shared_instance_metadata(aligned)
    misaligned = {**aligned, "shuffled_earlier_time_labels": ("different-invented-instance", False)}
    with pytest.raises(ContractError, match="Patient instance or outcome changed"):
        assert_shared_instance_metadata(misaligned)


def test_symmetric_gap_reversal_is_marked_ineligible_not_fabricated() -> None:
    prediction = dt.datetime(2024, 1, 1)
    events = tuple(ClinicalEvent(prediction - dt.timedelta(days=day), f"SYN/{day}") for day in (30, 20, 10))
    conditions = build_full_history_conditions(
        build_calendar_blocks(events, prediction), seed=3, instance_key="invented"
    )
    reversed_condition = conditions["reversed_earlier_gaps"]
    assert not reversed_condition.eligible
    assert "symmetric" in reversed_condition.ineligibility_reason


def test_latest_window_is_separate_information_reduction() -> None:
    prediction = dt.datetime(2024, 1, 1, 12)
    blocks = build_calendar_blocks(_events(prediction), prediction)
    reduced = latest_window_only(blocks, window_days=30)
    assert len(reduced) == 1
    assert len(clinical_signature(reduced)) < len(clinical_signature(blocks))


def test_descriptions_have_stable_code_fallback_and_synthetic_rendering() -> None:
    assert describe_code("SYN/A", {"SYN/A": "Invented finding"}) == "Invented finding [SYN/A]"
    assert describe_code("SYN/MISSING", {}) == "SYN/MISSING"
    prediction = dt.datetime(2024, 1, 1, 12)
    condition = build_full_history_conditions(
        build_calendar_blocks(_events(prediction), prediction), seed=1, instance_key="invented"
    )["correct_compact_time"]
    rendered = render_condition(condition, {"SYN/A": "Invented finding"})
    assert "Invented finding [SYN/A]" in rendered
    assert "[TIME 10 DAYS BEFORE]" in rendered


def test_packer_is_deterministic_and_truncates_long_synthetic_history() -> None:
    prediction = dt.datetime(2024, 1, 1)
    events = tuple(ClinicalEvent(prediction - dt.timedelta(days=day), f"SYN/{day}") for day in range(1, 30))
    condition = build_full_history_conditions(
        build_calendar_blocks(events, prediction), seed=9, instance_key="invented"
    )["correct_compact_time"]
    tokenizer = WhitespaceTokenizer()
    first = pack_history(condition, {}, tokenizer=tokenizer, budget=24)
    second = pack_history(condition, {}, tokenizer=tokenizer, budget=24)
    assert first == second
    assert first.truncated
    assert first.token_count <= 24
    assert 0 < first.included_event_count < first.total_event_count
    assert whitespace_estimate_for_synthetic_demo(first.text) == first.token_count
