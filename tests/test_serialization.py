from __future__ import annotations

import datetime as dt
from itertools import pairwise

import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.blocks import (
    ClinicalEvent,
    PredictionWindow,
    build_prediction_windows,
    clinical_signature,
    deduplicate_events,
    latest_code_occurrence,
    latest_window_only,
)
from temporal_ehr.serialization.conditions import (
    FULL_HISTORY_CONDITIONS,
    assert_condition_invariants,
    assert_shared_instance_metadata,
    build_full_history_conditions,
)
from temporal_ehr.serialization.packing import (
    PackingCache,
    content_matched_primary,
    independent_operational,
    pack_history,
)
from temporal_ehr.serialization.render import describe_code, render_condition

SEEDS = (20260815, 20260816, 20260817)


class WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))


class CountingWhitespaceTokenizer(WhitespaceTokenizer):
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        self.calls += 1
        return super().encode(text, add_special_tokens=add_special_tokens)


def _window(index: int, prediction: dt.datetime, count: int = 1) -> PredictionWindow:
    events = tuple(
        ClinicalEvent(prediction - dt.timedelta(days=index * 30 + 2, seconds=offset), f"SYN/{index}/{offset}")
        for offset in range(count - 1, -1, -1)
    )
    return PredictionWindow(index, 30, events)


def _irregular_windows(prediction: dt.datetime, count: int = 1) -> tuple[PredictionWindow, ...]:
    return tuple(_window(index, prediction, count) for index in (7, 4, 2, 0))


def test_exact_30_day_boundaries_and_time_of_day() -> None:
    prediction = dt.datetime(2024, 7, 1, 15, 30, 45)
    offsets = (
        (dt.timedelta(microseconds=1), "near-T", 0),
        (dt.timedelta(days=30), "exact-30", 0),
        (dt.timedelta(days=30, microseconds=1), "after-30", 1),
        (dt.timedelta(days=60), "exact-60", 1),
        (dt.timedelta(days=90), "exact-90", 2),
        (dt.timedelta(days=180), "exact-180", 5),
        (dt.timedelta(days=60, hours=3), "time-of-day", 2),
    )
    events = tuple(ClinicalEvent(prediction - elapsed, code) for elapsed, code, _ in offsets)
    windows = build_prediction_windows(events, prediction, window_days=30)
    observed = {event.code: window.window_index for window in windows for event in window.events}
    assert observed == {code: expected for _, code, expected in offsets}


def test_exact_90_day_boundaries_and_prediction_time_rejection() -> None:
    prediction = dt.datetime(2024, 7, 1, 15, 30)
    events = (
        ClinicalEvent(prediction - dt.timedelta(days=90), "SYN/90"),
        ClinicalEvent(prediction - dt.timedelta(days=90, microseconds=1), "SYN/90-plus"),
        ClinicalEvent(prediction - dt.timedelta(days=180), "SYN/180"),
    )
    windows = build_prediction_windows(events, prediction, window_days=90)
    observed = {event.code: window.window_index for window in windows for event in window.events}
    assert observed == {"SYN/90": 0, "SYN/90-plus": 1, "SYN/180": 1}
    with pytest.raises(ContractError, match="at or after"):
        build_prediction_windows((ClinicalEvent(prediction, "SYN/AT-T"),), prediction, window_days=90)


def test_exact_deduplication_preserves_recurrence_and_concurrency() -> None:
    prediction = dt.datetime(2024, 1, 1)
    event = ClinicalEvent(prediction - dt.timedelta(days=10), "SYN/R", 1.0, subject_id=7)
    events = (
        event,
        event,
        ClinicalEvent(event.time, "SYN/R", 2.0, subject_id=7),
        ClinicalEvent(event.time, "SYN/A", subject_id=7),
        ClinicalEvent(prediction - dt.timedelta(days=5), "SYN/R", 1.0, subject_id=7),
    )
    deduplicated, removed = deduplicate_events(events)
    assert removed == 1
    assert len(deduplicated) == 4
    assert sum(item.code == "SYN/R" for item in deduplicated) == 3
    windows = build_prediction_windows(deduplicated, prediction, window_days=30)
    same_time = [item for item in windows[0].events if item.time == event.time]
    assert [(item.code, item.numeric_value) for item in same_time] == [
        ("SYN/A", None),
        ("SYN/R", 1.0),
        ("SYN/R", 2.0),
    ]


def test_c0_c3_invariants_and_c2_gap_reversal() -> None:
    prediction = dt.datetime(2024, 1, 1)
    windows = _irregular_windows(prediction)
    conditions = build_full_history_conditions(windows, seed=SEEDS[0], instance_key="invented")
    assert tuple(conditions) == FULL_HISTORY_CONDITIONS
    assert_condition_invariants(conditions)
    assert all(
        clinical_signature(tuple(item.block for item in condition.blocks)) == clinical_signature(windows)
        for condition in conditions.values()
    )
    correct = tuple(item.label_window_index for item in conditions["C1_correct_compact_time"].blocks)
    reversed_labels = tuple(item.label_window_index for item in conditions["C2_reversed_earlier_gaps"].blocks)
    assert correct == (7, 4, 2, 0)
    assert reversed_labels == (7, 5, 3, 0)
    assert sorted(a - b for a, b in pairwise(correct)) == sorted(
        a - b for a, b in pairwise(reversed_labels)
    )
    assert reversed_labels[0] - reversed_labels[-1] == correct[0] - correct[-1]


def test_three_c3_seeds_are_deterministic_changed_and_content_preserving() -> None:
    prediction = dt.datetime(2024, 1, 1)
    windows = _irregular_windows(prediction)
    for seed in SEEDS:
        first = build_full_history_conditions(windows, seed=seed, instance_key="invented")
        second = build_full_history_conditions(windows, seed=seed, instance_key="invented")
        assert first == second
        condition = first["C3_shuffled_earlier_time_labels"]
        assert condition.eligible
        labels = tuple(item.label_window_index for item in condition.blocks)
        correct = tuple(window.window_index for window in windows)
        assert labels[:-1] != correct[:-1]
        assert sorted(labels[:-1]) == sorted(correct[:-1])
        assert labels[-1] == correct[-1]
        assert clinical_signature(tuple(item.block for item in condition.blocks)) == clinical_signature(windows)


def test_ineligible_falsifications_are_not_fabricated() -> None:
    prediction = dt.datetime(2024, 1, 1)
    short = build_full_history_conditions(_irregular_windows(prediction)[:2], seed=SEEDS[0], instance_key="short")
    assert not short["C2_reversed_earlier_gaps"].eligible
    assert not short["C3_shuffled_earlier_time_labels"].eligible
    symmetric_windows = tuple(_window(index, prediction) for index in (4, 2, 0))
    symmetric = build_full_history_conditions(symmetric_windows, seed=SEEDS[0], instance_key="symmetric")
    assert not symmetric["C2_reversed_earlier_gaps"].eligible
    assert "would not change" in str(symmetric["C2_reversed_earlier_gaps"].ineligibility_reason)


def test_range_labels_and_code_fallback_contain_no_timestamps() -> None:
    prediction = dt.datetime(2024, 1, 1)
    condition = build_full_history_conditions(
        _irregular_windows(prediction), seed=SEEDS[0], instance_key="invented"
    )["C1_correct_compact_time"]
    rendered = render_condition(condition, {"SYN/0/0": "Invented finding"})
    assert "[TIME=1_30_DAYS_BEFORE_T]" in rendered
    assert "[TIME=211_240_DAYS_BEFORE_T]" in rendered
    assert "2024" not in rendered
    assert describe_code("SYN/MISSING", {}) == "SYN/MISSING"


@pytest.mark.parametrize("budget", [4096, 8192])
def test_content_matched_primary_uses_identical_suffix_at_protocol_budgets(budget: int) -> None:
    prediction = dt.datetime(2024, 1, 1)
    conditions = build_full_history_conditions(
        _irregular_windows(prediction, count=800), seed=SEEDS[0], instance_key="long-synthetic"
    )
    packed = content_matched_primary(conditions, {}, tokenizer=WhitespaceTokenizer(), budget=budget)
    assert set(packed.histories) == set(FULL_HISTORY_CONDITIONS)
    assert len({history.retained_signature for history in packed.histories.values()}) == 1
    assert len({history.retained_block_signature for history in packed.histories.values()}) == 1
    assert all(history.token_count <= budget for history in packed.histories.values())


def test_independent_packing_reports_time_token_displacement() -> None:
    prediction = dt.datetime(2024, 1, 1)
    conditions = build_full_history_conditions(
        _irregular_windows(prediction, count=20), seed=SEEDS[0], instance_key="packing"
    )
    result = independent_operational(conditions, {}, tokenizer=CharacterTokenizer(), budget=600)
    assert result.displaced_events["C0_order_only"] == 0
    assert result.displaced_events["C1_correct_compact_time"] > 0
    for name, history in result.histories.items():
        assert result.displaced_events[name] == max(
            0, result.histories["C0_order_only"].included_event_count - history.included_event_count
        )


def test_shared_packing_cache_avoids_duplicate_policy_tokenization() -> None:
    prediction = dt.datetime(2024, 1, 1)
    descriptions: dict[str, str] = {}
    tokenizer = CountingWhitespaceTokenizer()
    conditions = build_full_history_conditions(
        _irregular_windows(prediction, count=800), seed=SEEDS[0], instance_key="cached-packing"
    )
    cache = PackingCache(descriptions, tokenizer)
    content_matched_primary(
        conditions,
        descriptions,
        tokenizer=tokenizer,
        budget=4096,
        cache=cache,
    )
    independent_operational(
        conditions,
        descriptions,
        tokenizer=tokenizer,
        budget=4096,
        cache=cache,
    )
    assert tokenizer.calls < 70


def test_information_reduction_baselines_preserve_order_and_correct_final_window() -> None:
    prediction = dt.datetime(2024, 1, 1)
    recurrent = ClinicalEvent(prediction - dt.timedelta(days=220), "SYN/RECURRENT")
    windows = (
        PredictionWindow(7, 30, (recurrent,)),
        PredictionWindow(2, 30, (ClinicalEvent(prediction - dt.timedelta(days=70), "SYN/UNIQUE"),)),
        PredictionWindow(0, 30, (ClinicalEvent(prediction - dt.timedelta(days=5), "SYN/RECURRENT"),)),
    )
    latest_window = latest_window_only(windows)
    latest_code = latest_code_occurrence(windows)
    assert latest_window == (windows[-1],)
    assert [event.code for block in latest_code for event in block.events] == ["SYN/UNIQUE", "SYN/RECURRENT"]
    for retained in (latest_window, latest_code):
        condition = build_full_history_conditions(retained, seed=SEEDS[0], instance_key="baseline")[
            "C1_correct_compact_time"
        ]
        assert condition.blocks[-1].label_window_index == retained[-1].window_index


def test_metadata_alignment_and_independent_packer_determinism() -> None:
    aligned = {name: ("synthetic_task", "invented-instance", True) for name in FULL_HISTORY_CONDITIONS}
    assert_shared_instance_metadata(aligned)
    with pytest.raises(ContractError, match="outcome changed"):
        assert_shared_instance_metadata({**aligned, "C3_shuffled_earlier_time_labels": ("task", "other", False)})
    prediction = dt.datetime(2024, 1, 1)
    condition = build_full_history_conditions(
        _irregular_windows(prediction, count=10), seed=SEEDS[0], instance_key="packing"
    )["C1_correct_compact_time"]
    first = pack_history(condition, {}, tokenizer=WhitespaceTokenizer(), budget=20)
    second = pack_history(condition, {}, tokenizer=WhitespaceTokenizer(), budget=20)
    assert first == second
    assert first.truncated and first.token_count <= 20
