"""Versioned bounded-omission M2 contract after canary job 1551.

The instruction model chooses optional singleton source-series omissions. Clinical facts are
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

VERSION = "m2_ordered_selection_v4"
PROMPT = """Select limited omissions from this pre-prediction record.
Return exactly one object keyed by the supplied window keys. Each value is an array of
source fact indices to OMIT, not retain. An empty array keeps every fact in that window.
Choose omissions only from omittable_indices and never exceed max_omissions. Prefer empty
arrays unless a listed singleton code-only fact is redundant for faithfully representing
the supplied history. Values, text, units, recurrent facts and ordered measurement trajectories
are protected and cannot be omitted. Never return clinical text, values, diagnoses or explanations.
All unlisted facts and all original window labels stay exactly as supplied. Do not repair time.
Output only the required JSON. Each array must be strictly increasing without duplicates."""


def omission_limits(source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """A conservative subset of the existing admissible retention region.

    Each window keeps >=90% of its unique code/source/unit concepts. All features
    contributing to the other five denominators are protected. The unchanged
    independent validator remains authoritative, including the actual token cap.
    """
    check_shape(source)
    result = {}
    for index, window in enumerate(source["windows"]):
        candidates = [
            i for i, fact in enumerate(window["facts"])
            if fact["unit"] is None
            and len(fact["sequence"]) == 1
            and fact["sequence"][0] == {"value": None, "text": None, "count": 1}
        ]
        result[f"w{index}"] = {
            "omittable_indices": candidates,
            "max_omissions": min(len(candidates), len(window["facts"]) // 10),
        }
    return result


def indexed_source(source: dict[str, Any]) -> dict[str, Any]:
    check_shape(source)
    limits = omission_limits(source)
    return {"windows": [
        {"key": f"w{i}", "label": window["label"], **limits[f"w{i}"],
         "facts": [{"fact_index": j, **copy.deepcopy(fact)}
                   for j, fact in enumerate(window["facts"])]}
        for i, window in enumerate(source["windows"])
    ]}


def response_schema(source: dict[str, Any]) -> dict[str, Any]:
    """Source-specific bounds are enforced during decoding and again locally."""
    properties = {}
    for key, limit in omission_limits(source).items():
        indices = limit["omittable_indices"]
        properties[key] = {
            "type": "array",
            "items": {"type": "integer", "enum": indices or [0]},
            "maxItems": limit["max_omissions"],
        }
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def payload(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": config["instruction_model"],
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps(
                indexed_source(source), sort_keys=True, ensure_ascii=False,
                separators=(",", ":"), allow_nan=False)},
        ],
        **copy.deepcopy(config["decoding"]),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": VERSION, "strict": True, "schema": response_schema(source)}},
        "stream": False,
    }


def identity(item: dict[str, Any], config: dict[str, Any], served_identity: str) -> str:
    return stable_hash({
        "version": VERSION,
        "request": payload(item["source"], config),
        "condition": item["condition"],
        "served_identity": served_identity,
        "attempt_policy": "two_fixed_requests_first_valid_no_feedback",
        "materialization": "source_exact_bounded_omission_selection_v4",
    })


def materialize(selection: Any, source: dict[str, Any]) -> dict[str, Any]:
    """Interpret only the exact model-returned omission plan; never repair it."""
    check_shape(source)
    limits = omission_limits(source)
    if not isinstance(selection, dict) or set(selection) != set(limits):
        raise SummaryContractError("selection_window_coverage")
    document = {"windows": []}
    for i, window in enumerate(source["windows"]):
        key = f"w{i}"
        omitted = selection[key]
        limit = limits[key]
        if not isinstance(omitted, list) or any(type(v) is not int for v in omitted):
            raise SummaryContractError("selection_index_type")
        if omitted != sorted(set(omitted)):
            raise SummaryContractError("selection_index_order_or_duplicate")
        if len(omitted) > limit["max_omissions"]:
            raise SummaryContractError("selection_omission_limit")
        if not set(omitted) <= set(limit["omittable_indices"]):
            raise SummaryContractError("selection_protected_or_invalid_index")
        document["windows"].append({
            "label": window["label"],
            "facts": [copy.deepcopy(fact) for j, fact in enumerate(window["facts"]) if j not in omitted],
        })
    check_shape(document)
    return document


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
        raise old.ContractError("Invalid v4 source/request/admission binding.")
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
        raise old.ContractError("Earlier valid v4 selection was skipped.")
    return result


def full_selection(source: dict[str, Any]) -> dict[str, Any]:
    """Synthetic-test helper, never a production fallback."""
    return {key: [] for key in omission_limits(source)}
