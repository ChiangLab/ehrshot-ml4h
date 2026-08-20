"""Private integrity manifests that expose hashes, never patient content."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.blocks import ClinicalEvent, exact_duplicate_key


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, float) and value != value:
        return "NaN"
    return value


def stable_hash(value: Any) -> str:
    """Hash a canonical JSON representation for deterministic drift detection."""
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def event_sequence_hash(events: Sequence[ClinicalEvent]) -> str:
    return stable_hash([exact_duplicate_key(event) for event in events])


def build_instance_manifest(
    *,
    dataset_revision: str,
    config_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    task: str,
    split: str,
    outcome: bool,
    subject_id: int,
    instance_key: str,
    prediction_time: dt.datetime,
    source_events: Sequence[ClinicalEvent],
    deduplicated_events: Sequence[ClinicalEvent],
    retained_signature: Sequence[tuple[object, ...]],
    retained_block_signature: Sequence[tuple[object, ...]],
    rendered_text: str,
    condition: str,
    perturbation_seed: int | None,
    comparison_seed: int,
    token_budget: int,
) -> dict[str, Any]:
    """Build one hash-only manifest record for the governed private output tree."""
    return {
        "manifest_version": 1,
        "dataset_revision_hash": stable_hash(dataset_revision),
        "config_revision_hash": stable_hash(config_revision),
        "model_revision_hash": stable_hash(model_revision),
        "tokenizer_revision_hash": stable_hash(tokenizer_revision),
        "task": task,
        "canonical_split": split,
        "outcome_hash": stable_hash(outcome),
        "patient_hash": stable_hash(subject_id),
        "prediction_instance_hash": stable_hash((instance_key, prediction_time)),
        "source_events_hash": event_sequence_hash(source_events),
        "deduplicated_events_hash": event_sequence_hash(deduplicated_events),
        "retained_content_hash": stable_hash(retained_signature),
        "retained_exact_event_hashes": [stable_hash(item) for item in retained_signature],
        "retained_block_content_hash": stable_hash(retained_block_signature),
        "rendered_text_hash": stable_hash(rendered_text),
        "temporal_condition": condition,
        "perturbation_seed": perturbation_seed,
        "comparison_seed": comparison_seed,
        "token_budget": token_budget,
    }


def require_private_output(path: Path, repository_root: Path | None = None) -> Path:
    """Fail closed unless a patient-level artifact is under outputs/private/."""
    root = (repository_root or Path.cwd()).resolve()
    private_root = (root / "outputs/private").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(private_root):
        raise ContractError("Patient-level manifests must be written under outputs/private/.")
    return resolved


def require_aggregate_output(path: Path, repository_root: Path | None = None) -> Path:
    """Fail closed unless an aggregate review artifact is under reports/generated/."""
    root = (repository_root or Path.cwd()).resolve()
    report_root = (root / "reports/generated").resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(report_root):
        raise ContractError("Aggregate review reports must be written under reports/generated/.")
    return resolved


def append_private_manifest(path: Path, record: Mapping[str, Any], repository_root: Path | None = None) -> None:
    destination = require_private_output(path, repository_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
