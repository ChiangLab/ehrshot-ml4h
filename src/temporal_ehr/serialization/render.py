"""Stable rendering of structured codes and temporal block labels."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from temporal_ehr.data.zip_assets import read_parquet_member
from temporal_ehr.serialization.blocks import ClinicalEvent
from temporal_ehr.serialization.conditions import LabeledBlock, TemporalCondition


def load_code_descriptions(asset_root: Path) -> dict[str, str]:
    """Load official downloaded descriptions, falling back later to the code string."""
    table = read_parquet_member(asset_root, "metadata/codes.parquet", ["code", "description"])
    return {
        str(code): str(description)
        for code, description in zip(table["code"].to_pylist(), table["description"].to_pylist(), strict=True)
        if description
    }


def describe_code(code: str, descriptions: Mapping[str, str]) -> str:
    description = descriptions.get(code)
    return f"{description} [{code}]" if description else code


def render_event(event: ClinicalEvent, descriptions: Mapping[str, str]) -> str:
    """Render every structured field deterministically; no clinical ordering is inferred."""
    parts = [describe_code(event.code, descriptions)]
    if event.numeric_value is not None:
        parts.append(f"value={event.numeric_value:g}")
    if event.text_value is not None:
        parts.append(f"text_value={event.text_value}")
    if event.unit is not None:
        parts.append(f"unit={event.unit}")
    if event.omop_table is not None:
        parts.append(f"source={event.omop_table}")
    return " | ".join(parts)


def render_labeled_block(block: LabeledBlock, descriptions: Mapping[str, str]) -> tuple[str, ...]:
    if block.label_window_index is None:
        label = "[CLINICAL_BLOCK]"
    else:
        start = block.label_window_index * block.block.window_days + 1
        end = (block.label_window_index + 1) * block.block.window_days
        label = f"[TIME={start}_{end}_DAYS_BEFORE_T]"
    return (label, *(f"- {render_event(event, descriptions)}" for event in block.block.events))


def render_condition(condition: TemporalCondition, descriptions: Mapping[str, str]) -> str:
    """Render an eligible condition without dates, timestamps, or patient identifiers."""
    if not condition.eligible:
        raise ValueError(f"Condition {condition.name} is ineligible: {condition.ineligibility_reason}")
    return "\n".join(line for block in condition.blocks for line in render_labeled_block(block, descriptions))
