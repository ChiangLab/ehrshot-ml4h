"""Strictly censored, prediction-anchored calendar block construction."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from temporal_ehr.data.contracts import ContractError


@dataclass(frozen=True)
class ClinicalEvent:
    """One structured event; tuple order is used only as a deterministic tie-break."""

    time: dt.datetime
    code: str
    numeric_value: float | None = None
    text_value: str | None = None
    unit: str | None = None
    omop_table: str | None = None


@dataclass(frozen=True)
class CalendarBlock:
    """An atomic day or a prediction-anchored aggregate of day blocks."""

    relative_day: int
    window_days: int
    events: tuple[ClinicalEvent, ...]


def _event_sort_key(event: ClinicalEvent) -> tuple[Any, ...]:
    return (
        event.time,
        event.code,
        event.numeric_value is not None,
        event.numeric_value or 0.0,
        event.text_value or "",
        event.unit or "",
        event.omop_table or "",
    )


def event_from_mapping(row: Mapping[str, Any]) -> ClinicalEvent:
    """Convert a MEDS-style mapping without altering structured event content."""
    return ClinicalEvent(
        time=row["time"],
        code=str(row["code"]),
        numeric_value=row.get("numeric_value"),
        text_value=row.get("text_value"),
        unit=row.get("unit"),
        omop_table=row.get("omop_table"),
    )


def build_calendar_blocks(
    events: Sequence[ClinicalEvent],
    prediction_time: dt.datetime,
    *,
    window_days: int = 1,
) -> tuple[CalendarBlock, ...]:
    """Create chronological blocks after enforcing event_time < prediction_time."""
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    grouped: defaultdict[int, list[ClinicalEvent]] = defaultdict(list)
    for event in events:
        if event.time >= prediction_time:
            raise ContractError("History construction received an event at or after prediction time.")
        days_before = (prediction_time.date() - event.time.date()).days
        bucket = days_before // window_days
        grouped[bucket].append(event)
    blocks = [
        CalendarBlock(
            relative_day=bucket * window_days,
            window_days=window_days,
            events=tuple(sorted(block_events, key=_event_sort_key)),
        )
        for bucket, block_events in grouped.items()
    ]
    blocks.sort(key=lambda block: block.relative_day, reverse=True)
    return tuple(blocks)


def latest_window_only(
    blocks: Sequence[CalendarBlock],
    *,
    window_days: int,
) -> tuple[CalendarBlock, ...]:
    """Return an explicitly information-reduced recent-history baseline."""
    if window_days < 1:
        raise ValueError("window_days must be positive.")
    return tuple(block for block in blocks if block.relative_day < window_days)


def clinical_signature(blocks: Sequence[CalendarBlock]) -> tuple[tuple[Any, ...], ...]:
    """Return event content and order, excluding all temporal labels."""
    return tuple(
        (
            event.time,
            event.code,
            event.numeric_value,
            event.text_value,
            event.unit,
            event.omop_table,
        )
        for block in blocks
        for event in block.events
    )
