"""Deterministic recent-first packing under an exact tokenizer budget."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from temporal_ehr.serialization.conditions import TemporalCondition
from temporal_ehr.serialization.render import render_labeled_block


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


def load_qwen_tokenizer(model_path: Path) -> Tokenizer:
    """Load an already-downloaded tokenizer; this function never downloads a model."""
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
            "model download is intentionally outside this notebook pack."
        )
    try:
        return AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the local Qwen tokenizer. Verify QWEN_EMBEDDING_PATH and the pinned model assets."
        ) from exc


def _count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def pack_history(
    condition: TemporalCondition,
    descriptions: Mapping[str, str],
    *,
    tokenizer: Tokenizer,
    budget: int = 4_096,
) -> PackedHistory:
    """Keep the newest complete content that fits, with exact deterministic tokenization."""
    if not condition.eligible:
        raise ValueError(f"Cannot pack ineligible condition: {condition.ineligibility_reason}")
    if budget < 1:
        raise ValueError("Token budget must be positive.")
    accepted: list[str] = []
    included_events = 0
    total_events = sum(len(item.block.events) for item in condition.blocks)
    for labeled_block in reversed(condition.blocks):
        lines = list(render_labeled_block(labeled_block, descriptions))
        candidate = "\n".join([*lines, *accepted])
        if _count(tokenizer, candidate) <= budget:
            accepted = [*lines, *accepted]
            included_events += len(labeled_block.block.events)
            continue

        header = lines[0]
        event_lines = lines[1:]
        low, high = 1, len(event_lines)
        best: list[str] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            partial = [header, *event_lines[-midpoint:]]
            partial_candidate = "\n".join([*partial, *accepted])
            if _count(tokenizer, partial_candidate) <= budget:
                best = partial
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best is not None:
            accepted = [*best, *accepted]
            included_events += max(0, len(best) - 1)
        break
    text = "\n".join(accepted)
    token_count = _count(tokenizer, text) if text else 0
    return PackedHistory(text, token_count, budget, included_events, total_events, included_events < total_events)


def whitespace_estimate_for_synthetic_demo(text: str) -> int:
    """Return a whitespace estimate; this is explicitly not a tokenizer token count."""
    return len(text.split())
