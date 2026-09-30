"""Versioned extractive M2 selection contract after production job 993.

The instruction model chooses ordered source-series positions. Clinical facts are
materialized only from the immutable source, so generated content can never alter a
code, value, unit, recurrence count, time label, or series order.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_render_v2 as renderer
from temporal_ehr.final_m23.structured_summary import SummaryContractError, check_shape
from temporal_ehr.serialization.integrity import stable_hash

VERSION = "m2_ordered_selection_v3"
PROMPT = """Select a faithful bounded summary of the supplied pre-prediction record.
The input has ordered windows and, within each window, ordered clinical fact series. Each
window and fact has an integer index. Return one selection entry for every supplied window,
in the same window-index order. Each fact_indices array must contain unique strictly increasing
source fact indices. Select facts only; never emit clinical codes, values, units, counts, labels,
explanations, diagnoses, markdown, or additional fields.

Retain at least 90% of window-specific concepts, 95% of numeric/text-value and unit
associations, and 80% of recurrence and directional series. Prefer retaining all facts when the
resulting deterministic rendering fits the downstream context; otherwise omit redundant whole
series while meeting every retention floor. Never select only part of a fact series. Preserve all
windows, including windows with no selected facts. Output only the required JSON object."""


def indexed_source(source: dict[str, Any]) -> dict[str, Any]:
    """Add positional handles without changing any clinical array order or value."""
    check_shape(source)
    windows = []
    for window_index, window in enumerate(source["windows"]):
        facts = []
        for fact_index, fact in enumerate(window["facts"]):
            facts.append({"fact_index": fact_index, **copy.deepcopy(fact)})
        windows.append({
            "window_index": window_index,
            "label": window["label"],
            "facts": facts,
        })
    return {"windows": windows}


def response_schema() -> dict[str, Any]:
    """Transport schema; source-specific ordering/range checks remain mandatory."""
    return {
        "type": "object",
        "properties": {
            "windows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "window_index": {"type": "integer", "minimum": 0},
                        "fact_indices": {
                            "type": "array",
                            "items": {"type": "integer", "minimum": 0},
                        },
                    },
                    "required": ["window_index", "fact_indices"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["windows"],
        "additionalProperties": False,
    }


def payload(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    request = {
        "model": config["instruction_model"],
        "messages": [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    indexed_source(source),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            },
        ],
        **copy.deepcopy(config["decoding"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": VERSION,
                "strict": True,
                "schema": response_schema(),
            },
        },
        "stream": False,
    }
    return request


def identity(item: dict[str, Any], config: dict[str, Any], served_identity: str) -> str:
    return stable_hash({
        "version": VERSION,
        "request": payload(item["source"], config),
        "condition": item["condition"],
        "served_identity": served_identity,
        "attempt_policy": "two_fixed_requests_first_valid_no_feedback",
        "materialization": "source_exact_ordered_position_selection_v1",
    })


def materialize(selection: Any, source: dict[str, Any]) -> dict[str, Any]:
    """Build a clinical document exclusively from source positions."""
    check_shape(source)
    if not isinstance(selection, dict) or set(selection) != {"windows"}:
        raise SummaryContractError("selection_schema")
    windows = selection["windows"]
    if not isinstance(windows, list) or len(windows) != len(source["windows"]):
        raise SummaryContractError("selection_window_coverage")
    output = {"windows": []}
    for expected_index, (selected, original) in enumerate(
        zip(windows, source["windows"], strict=True)
    ):
        if (
            not isinstance(selected, dict)
            or set(selected) != {"window_index", "fact_indices"}
            or type(selected["window_index"]) is not int
            or selected["window_index"] != expected_index
            or not isinstance(selected["fact_indices"], list)
        ):
            raise SummaryContractError("selection_window_order")
        indices = selected["fact_indices"]
        if any(type(index) is not int for index in indices):
            raise SummaryContractError("selection_index_type")
        if indices != sorted(set(indices)):
            raise SummaryContractError("selection_index_order_or_duplicate")
        if any(index < 0 or index >= len(original["facts"]) for index in indices):
            raise SummaryContractError("selection_index_range")
        output["windows"].append({
            "label": original["label"],
            "facts": [copy.deepcopy(original["facts"][index]) for index in indices],
        })
    check_shape(output)
    return output


def validate(
    selection: Any,
    source: dict[str, Any],
    *,
    thresholds: dict[str, Any],
    token_counter: Any,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    document = materialize(selection, source)
    validation = renderer.validate(
        document,
        source,
        thresholds=thresholds,
        token_counter=token_counter,
    )
    summary = renderer.render(document)
    return document, validation, summary


def assess(
    item: dict[str, Any],
    config: dict[str, Any],
    thresholds: dict[str, Any],
    counter: Any,
    directory: Path,
    served_identity: str,
    tokenizer_hash: str,
    client: Any = None,
) -> dict[str, Any]:
    """Two same-contract attempts; never repair a rejected model selection."""
    directory.mkdir(parents=True, exist_ok=False)
    old.write_once(directory / "source.json", item)
    request = payload(item["source"], config)
    old.write_once(directory / "request.json", request)
    client = client or old.PrivateClient(config)
    errors: list[str] = []
    rejected: set[str] = set()
    for attempt in (1, 2):
        try:
            response = client.request("chat/completions", request)
            old.write_once(directory / f"response-{attempt}.json", response)
            selection = old.response_document(response)
            digest = stable_hash(selection)
            if digest in rejected:
                raise SummaryContractError("identical_rejected_selection")
            rejected.add(digest)
            document, validation, summary = validate(
                selection,
                item["source"],
                thresholds=thresholds,
                token_counter=counter,
            )
            key = identity(item, config, served_identity)
            result = {
                "status": "PASS",
                "selection": selection,
                "document": document,
                "summary": summary,
                "validation": validation,
                "attempts": attempt,
                "errors": errors,
                "generation_key": key,
                "embedding_input_key": renderer.embedding_identity(
                    key, summary, tokenizer_hash
                ),
            }
            old.write_once(directory / "assessment.json", result)
            return result
        except SummaryContractError as exc:
            errors.append(str(exc))
        except old.NonRetryableSummaryError:
            errors.append("embedding_token_budget")
        except RuntimeError:
            errors.append("endpoint_failure")
    result = {"status": "FAIL", "errors": errors}
    old.write_once(directory / "assessment.json", result)
    return result


def verify_item(
    directory: Path,
    item: dict[str, Any],
    config: dict[str, Any],
    thresholds: dict[str, Any],
    counter: Any,
    served_identity: str,
    tokenizer_hash: str,
) -> dict[str, Any]:
    result = old.read(directory / "assessment.json")
    attempt = result.get("attempts")
    if (
        result.get("status") != "PASS"
        or type(attempt) is not int
        or attempt not in (1, 2)
        or old.read(directory / "source.json") != item
        or old.read(directory / "request.json") != payload(item["source"], config)
    ):
        raise old.ContractError("Invalid v3 source/request/admission binding.")
    selection = old.response_document(old.read(directory / f"response-{attempt}.json"))
    document, validation, summary = validate(
        selection,
        item["source"],
        thresholds=thresholds,
        token_counter=counter,
    )
    key = identity(item, config, served_identity)
    if (
        selection != result["selection"]
        or document != result["document"]
        or validation != result["validation"]
        or summary != result["summary"]
        or result["generation_key"] != key
        or result["embedding_input_key"]
        != renderer.embedding_identity(key, summary, tokenizer_hash)
    ):
        raise old.ContractError("V3 response or deterministic materialization changed.")
    for previous in range(1, attempt):
        path = directory / f"response-{previous}.json"
        if not path.exists():
            continue
        try:
            earlier = old.response_document(old.read(path))
            validate(
                earlier,
                item["source"],
                thresholds=thresholds,
                token_counter=counter,
            )
        except (SummaryContractError, old.NonRetryableSummaryError):
            continue
        raise old.ContractError("Earlier valid v3 selection was skipped.")
    return result


def full_selection(source: dict[str, Any]) -> dict[str, Any]:
    """Synthetic-test helper, never a production fallback."""
    check_shape(source)
    return {
        "windows": [
            {
                "window_index": index,
                "fact_indices": list(range(len(window["facts"]))),
            }
            for index, window in enumerate(source["windows"])
        ]
    }
