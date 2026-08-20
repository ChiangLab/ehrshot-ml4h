"""Deterministic event deduplication and prediction-relative block construction."""

from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from temporal_ehr.data.contracts import ContractError


@dataclass(frozen=True)
class ClinicalEvent:
    """One structured event; subject_id is private metadata and is never rendered."""

    time: dt.datetime
    code: str
    numeric_value: float | None = None
    text_value: str | None = None
    unit: str | None = None
    omop_table: str | None = None
    subject_id: int | None = None


@dataclass(frozen=True)
class CalendarBlock:
    """A calendar-day QC block, not a primary experimental representation."""

    relative_day: int
    window_days: int
    events: tuple[ClinicalEvent, ...]


@dataclass(frozen=True)
class PredictionWindow:
    """A half-open prediction-relative window, stored oldest to newest."""

    window_index: int
    window_days: int
    events: tuple[ClinicalEvent, ...]

    @property
    def label_start_day(self) -> int:
        return self.window_index * self.window_days + 1

    @property
    def label_end_day(self) -> int:
        return (self.window_index + 1) * self.window_days


def _optional_float_key(value: float | None) -> tuple[int, int, float]:
    if value is None:
        return (0, 0, 0.0)
    if math.isnan(value):
        return (1, 1, 0.0)
    return (1, 0, value)


def exact_duplicate_key(event: ClinicalEvent) -> tuple[Any, ...]:
    """Return the seven-field EHRSHOT row identity used only for exact deduplication."""
    return (
        event.subject_id,
        event.time,
        event.code,
        _optional_float_key(event.numeric_value),
        event.text_value,
        event.unit,
        event.omop_table,
    )


def event_sort_key(event: ClinicalEvent) -> tuple[Any, ...]:
    """Stable order; fields after time are tie-breakers, not inferred clinical order."""
    return (
        event.time,
        event.code,
        _optional_float_key(event.numeric_value),
        event.text_value is not None,
        event.text_value or "",
        event.unit is not None,
        event.unit or "",
        event.omop_table is not None,
        event.omop_table or "",
        event.subject_id if event.subject_id is not None else -1,
    )


def event_from_mapping(row: Mapping[str, Any]) -> ClinicalEvent:
    """Convert a MEDS-style mapping without altering structured event content."""
    subject_id = row.get("subject_id")
    return ClinicalEvent(
        time=row["time"],
        code=str(row["code"]),
        numeric_value=row.get("numeric_value"),
        text_value=row.get("text_value"),
        unit=row.get("unit"),
        omop_table=row.get("omop_table"),
        subject_id=int(subject_id) if subject_id is not None else None,
    )


def deduplicate_events(events: Sequence[ClinicalEvent]) -> tuple[tuple[ClinicalEvent, ...], int]:
    """Remove exact seven-field duplicate rows while preserving clinical recurrence."""
    unique: list[ClinicalEvent] = []
    seen: set[tuple[Any, ...]] = set()
    for event in sorted(events, key=event_sort_key):
        key = exact_duplicate_key(event)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return tuple(unique), len(events) - len(unique)


def build_calendar_blocks(
    events: Sequence[ClinicalEvent],
    prediction_time: dt.datetime,
    *,
    window_days: int = 1,
) -> tuple[CalendarBlock, ...]:
    """Create calendar-date blocks for QC after strict censoring."""
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    grouped: defaultdict[int, list[ClinicalEvent]] = defaultdict(list)
    for event in events:
        if event.time >= prediction_time:
            raise ContractError("History construction received an event at or after prediction time.")
        days_before = (prediction_time.date() - event.time.date()).days
        grouped[days_before // window_days].append(event)
    blocks = [
        CalendarBlock(
            relative_day=bucket * window_days,
            window_days=window_days,
            events=tuple(sorted(block_events, key=event_sort_key)),
        )
        for bucket, block_events in grouped.items()
    ]
    blocks.sort(key=lambda block: block.relative_day, reverse=True)
    return tuple(blocks)


def build_prediction_windows(
    events: Sequence[ClinicalEvent],
    prediction_time: dt.datetime,
    *,
    window_days: int,
) -> tuple[PredictionWindow, ...]:
    """Build exact windows [T-(i+1)w, T-iw), excluding T itself."""
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    width = dt.timedelta(days=window_days)
    grouped: defaultdict[int, list[ClinicalEvent]] = defaultdict(list)
    for event in events:
        elapsed = prediction_time - event.time
        if elapsed <= dt.timedelta(0):
            raise ContractError("History construction received an event at or after prediction time.")
        # Python datetimes have microsecond resolution. Subtracting one microsecond makes
        # exact 30/60/90-day boundaries belong to the newer interval exactly once.
        window_index = (elapsed - dt.timedelta(microseconds=1)) // width
        grouped[int(window_index)].append(event)
    windows = [
        PredictionWindow(
            window_index=index,
            window_days=window_days,
            events=tuple(sorted(window_events, key=event_sort_key)),
        )
        for index, window_events in grouped.items()
    ]
    windows.sort(key=lambda block: block.window_index, reverse=True)
    return tuple(windows)


def latest_window_only(windows: Sequence[PredictionWindow]) -> tuple[PredictionWindow, ...]:
    """Retain the entire most recent nonempty configured observation window."""
    return (tuple(windows)[-1],) if windows else ()


def latest_code_occurrence(windows: Sequence[PredictionWindow]) -> tuple[PredictionWindow, ...]:
    """Retain one deterministic latest occurrence per code, preserving chronology."""
    latest: dict[str, ClinicalEvent] = {}
    for window in windows:
        for event in window.events:
            latest[event.code] = event
    retained_keys = {exact_duplicate_key(event) for event in latest.values()}
    result = tuple(
        PredictionWindow(
            window.window_index,
            window.window_days,
            tuple(event for event in window.events if exact_duplicate_key(event) in retained_keys),
        )
        for window in windows
    )
    return tuple(window for window in result if window.events)


def clinical_signature(
    blocks: Sequence[CalendarBlock | PredictionWindow],
) -> tuple[tuple[Any, ...], ...]:
    """Return exact ordered event identities, excluding temporal labels."""
    return tuple(exact_duplicate_key(event) for block in blocks for event in block.events)


def block_content_signature(
    blocks: Sequence[PredictionWindow],
) -> tuple[tuple[int, tuple[tuple[Any, ...], ...]], ...]:
    """Return window order/boundaries and exact event order without rendered labels."""
    return tuple(
        (block.window_index, tuple(exact_duplicate_key(event) for event in block.events))
        for block in blocks
    )
