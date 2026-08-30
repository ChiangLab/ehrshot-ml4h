"""Content-matched primary and independently packed operational policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.blocks import PredictionWindow, block_content_signature, clinical_signature
from temporal_ehr.serialization.conditions import LabeledBlock, TemporalCondition
from temporal_ehr.serialization.render import render_condition


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> Sequence[int]: ...


@dataclass(frozen=True)
class PackedHistory:
    text: str
    token_count: int
    budget: int
    included_event_count: int
    total_event_count: int
    truncated: bool
    retained_signature: tuple[tuple[object, ...], ...]
    retained_block_signature: tuple[tuple[int, tuple[tuple[object, ...], ...]], ...]


@dataclass(frozen=True)
class ContentMatchedPacking:
    policy: str
    budget: int
    histories: dict[str, PackedHistory]


@dataclass(frozen=True)
class IndependentPacking:
    policy: str
    budget: int
    histories: dict[str, PackedHistory]
    displaced_events: dict[str, int]


@dataclass(frozen=True)
class _EvaluatedSuffix:
    condition: TemporalCondition
    text: str
    token_count: int
    retained_signature: tuple[tuple[object, ...], ...]
    retained_block_signature: tuple[tuple[int, tuple[tuple[object, ...], ...]], ...]


class PackingCache:
    """Per-instance cache for exact render/tokenizer evaluations."""

    def __init__(self, descriptions: Mapping[str, str], tokenizer: Tokenizer) -> None:
        self.descriptions = descriptions
        self.tokenizer = tokenizer
        self._evaluations: dict[tuple[int, int], _EvaluatedSuffix] = {}
        self._condition_references: dict[int, TemporalCondition] = {}

    def evaluate(self, condition: TemporalCondition, event_count: int) -> _EvaluatedSuffix:
        condition_id = id(condition)
        existing = self._condition_references.setdefault(condition_id, condition)
        if existing is not condition:
            raise RuntimeError("Packing-cache condition identity collision.")
        key = (condition_id, event_count)
        if key not in self._evaluations:
            suffix = _suffix_condition(condition, event_count)
            text = render_condition(suffix, self.descriptions) if suffix.blocks else ""
            blocks = tuple(item.block for item in suffix.blocks)
            self._evaluations[key] = _EvaluatedSuffix(
                suffix,
                text,
                _count(self.tokenizer, text),
                clinical_signature(blocks),
                block_content_signature(blocks),
            )
        return self._evaluations[key]


def load_qwen_tokenizer(model_path: Path) -> Tokenizer:
    """Load an already-downloaded tokenizer without network access."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Real-data packing requires transformers and an already-downloaded Qwen tokenizer. "
            "Install the model extras and set QWEN_EMBEDDING_PATH on the DGX."
        ) from exc
    if not model_path.exists():
        raise RuntimeError(
            "Real-data packing requires an already-downloaded Qwen tokenizer at QWEN_EMBEDDING_PATH; "
            "model download is intentionally outside this review."
        )
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the local Qwen tokenizer. Verify QWEN_EMBEDDING_PATH and pinned tokenizer assets."
        ) from exc


def _count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _suffix_condition(condition: TemporalCondition, event_count: int) -> TemporalCondition:
    if event_count < 0:
        raise ValueError("event_count cannot be negative.")
    total = sum(len(item.block.events) for item in condition.blocks)
    remaining = min(event_count, total)
    kept_reversed: list[LabeledBlock] = []
    for item in reversed(condition.blocks):
        if remaining <= 0:
            break
        keep = min(remaining, len(item.block.events))
        events = item.block.events[-keep:]
        kept_reversed.append(
            LabeledBlock(
                PredictionWindow(item.block.window_index, item.block.window_days, events),
                item.label_window_index,
            )
        )
        remaining -= keep
    return TemporalCondition(
        condition.name,
        tuple(reversed(kept_reversed)),
        condition.eligible,
        condition.perturbation_seed,
        condition.ineligibility_reason,
    )


def _packed_from_evaluation(
    evaluation: _EvaluatedSuffix,
    budget: int,
    total_events: int,
) -> PackedHistory:
    if evaluation.token_count > budget:
        raise ContractError("Packing produced text over the exact tokenizer budget.")
    return PackedHistory(
        evaluation.text,
        evaluation.token_count,
        budget,
        len(evaluation.retained_signature),
        total_events,
        len(evaluation.retained_signature) < total_events,
        evaluation.retained_signature,
        evaluation.retained_block_signature,
    )


def pack_history(
    condition: TemporalCondition,
    descriptions: Mapping[str, str],
    *,
    tokenizer: Tokenizer,
    budget: int = 4_096,
    cache: PackingCache | None = None,
) -> PackedHistory:
    """Independently retain the largest recent event suffix fitting one condition."""
    if not condition.eligible:
        raise ValueError(f"Cannot pack ineligible condition: {condition.ineligibility_reason}")
    if budget < 1:
        raise ValueError("Token budget must be positive.")
    packing_cache = cache or PackingCache(descriptions, tokenizer)
    if packing_cache.tokenizer is not tokenizer or packing_cache.descriptions is not descriptions:
        raise ValueError("PackingCache must use the same tokenizer and descriptions objects.")
    total = sum(len(item.block.events) for item in condition.blocks)
    low, high, best = 0, total, 0
    while low <= high:
        midpoint = (low + high) // 2
        if packing_cache.evaluate(condition, midpoint).token_count <= budget:
            best = midpoint
            low = midpoint + 1
        else:
            high = midpoint - 1
    return _packed_from_evaluation(packing_cache.evaluate(condition, best), budget, total)


def content_matched_primary(
    conditions: Mapping[str, TemporalCondition],
    descriptions: Mapping[str, str],
    *,
    tokenizer: Tokenizer,
    budget: int,
    cache: PackingCache | None = None,
) -> ContentMatchedPacking:
    """Pack one shared recent suffix sized by the most expensive eligible condition."""
    eligible = {name: condition for name, condition in conditions.items() if condition.eligible}
    if not eligible:
        raise ContractError("No temporal condition is eligible for content matching.")
    totals = {sum(len(item.block.events) for item in condition.blocks) for condition in eligible.values()}
    if len(totals) != 1:
        raise ContractError("Temporal conditions do not contain the same number of source events.")
    total = totals.pop()
    packing_cache = cache or PackingCache(descriptions, tokenizer)
    independent_limits = {
        name: pack_history(
            condition,
            descriptions,
            tokenizer=tokenizer,
            budget=budget,
            cache=packing_cache,
        ).included_event_count
        for name, condition in eligible.items()
    }
    best = min(independent_limits.values())
    histories = {
        name: _packed_from_evaluation(packing_cache.evaluate(condition, best), budget, total)
        for name, condition in eligible.items()
    }
    signatures = {history.retained_signature for history in histories.values()}
    block_signatures = {history.retained_block_signature for history in histories.values()}
    if len(signatures) != 1 or len(block_signatures) != 1:
        raise ContractError("content_matched_primary retained different clinical content across conditions.")
    return ContentMatchedPacking("content_matched_primary", budget, histories)


def independent_operational(
    conditions: Mapping[str, TemporalCondition],
    descriptions: Mapping[str, str],
    *,
    tokenizer: Tokenizer,
    budget: int,
    cache: PackingCache | None = None,
) -> IndependentPacking:
    """Independently pack eligible conditions and quantify evidence displacement from C0."""
    histories = {
        name: pack_history(
            condition,
            descriptions,
            tokenizer=tokenizer,
            budget=budget,
            cache=cache,
        )
        for name, condition in conditions.items()
        if condition.eligible
    }
    c0 = histories.get("C0_order_only")
    if c0 is None:
        raise ContractError("C0_order_only must be eligible for operational packing.")
    displacement = {
        name: max(0, c0.included_event_count - history.included_event_count)
        for name, history in histories.items()
    }
    return IndependentPacking("independent_operational", budget, histories, displacement)


def whitespace_estimate_for_synthetic_demo(text: str) -> int:
    """Return a whitespace estimate; this is explicitly not a Qwen token count."""
    return len(text.split())
