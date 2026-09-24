"""Versioned single-pass M2 contract; never used by historical M1/M2/M3 paths.

The LLM emits structured clinical facts, not fact IDs. Runs retain ordered repeated
measurements. Only whole, source-exact series may be omitted under retention gates.
This is a within-window grouped representation, not a lossless global event order.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from itertools import pairwise
from typing import Any

from temporal_ehr.serialization.integrity import stable_hash

VERSION = "m2_ordered_structured_v1"
TIME_LABEL_RE = re.compile(r"\[TIME=\d+_\d+_DAYS_BEFORE_T\]")
SOURCE_FIELD_RE = re.compile(r" \| (?=(?:value|text_value|unit|source)=)")
PROMPT = """Summarize the supplied pre-prediction record once using the required JSON schema.
Input and output have a windows array. Preserve every window label exactly, once, in supplied
array order, including intentionally falsified labels. Never sort labels or repair time.
Each fact is a code/source/unit series whose sequence contains ordered value/text/count runs.
Copy each retained fact and its complete sequence EXACTLY, including nulls, numeric strings,
text strings, counts, and repeated returns to earlier values. Do not combine series or reorder
facts. Do not expand codes, infer diagnoses, calculate trends, add commentary, or predict outcomes.
Compress by omitting only redundant whole facts while preserving at least 90% of window-specific
concepts, 95% of numeric/text-value and unit associations, and 80% of recurrence and directional
series. Aim to retain all facts when feasible. Never shorten an individual retained sequence.
Output only the JSON object, no markdown. Keep it concise; do not pretty-print.
An empty input windows array must yield an empty windows array."""


class SummaryContractError(ValueError):
    """Content-free, safe-to-report error code."""


def fail(code: str) -> None:
    raise SummaryContractError(code)


def strict_json(text: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                fail("duplicate_json_key")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: fail("nonfinite_json"))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise SummaryContractError("invalid_json") from exc


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def response_schema() -> dict[str, Any]:
    """vLLM JSON-schema transport; semantic checks below remain mandatory."""
    def obj(properties: dict[str, Any]) -> dict[str, Any]:
        return {"type": "object", "properties": properties, "required": list(properties),
                "additionalProperties": False}

    nullable = {"type": ["string", "null"]}
    run = obj({"value": nullable, "text": nullable, "count": {"type": "integer", "minimum": 1}})
    fact = obj({"code": {"type": "string", "minLength": 1}, "source": nullable, "unit": nullable,
                "sequence": {"type": "array", "items": run, "minItems": 1}})
    window = obj({"label": {"type": "string"}, "facts": {"type": "array", "items": fact}})
    return obj({"windows": {"type": "array", "items": window}})


def check_shape(document: Any) -> None:
    def keys(value: Any, expected: set[str]) -> None:
        if not isinstance(value, dict) or set(value) != expected:
            fail("schema_fields")

    keys(document, {"windows"})
    if not isinstance(document["windows"], list):
        fail("schema_windows")
    labels = set()
    for window in document["windows"]:
        keys(window, {"label", "facts"})
        label = window["label"]
        if not isinstance(label, str) or not TIME_LABEL_RE.fullmatch(label) or label in labels:
            fail("invalid_or_duplicate_window")
        labels.add(label)
        if not isinstance(window["facts"], list):
            fail("schema_facts")
        identities = set()
        for fact in window["facts"]:
            keys(fact, {"code", "source", "unit", "sequence"})
            if not isinstance(fact["code"], str) or not fact["code"].strip():
                fail("schema_code")
            for name in ("source", "unit"):
                if fact[name] is not None and not isinstance(fact[name], str):
                    fail("schema_string")
            identity = (fact["code"], fact["source"], fact["unit"])
            if identity in identities:
                fail("duplicate_series")
            identities.add(identity)
            if not isinstance(fact["sequence"], list) or not fact["sequence"]:
                fail("schema_sequence")
            previous = None
            for run in fact["sequence"]:
                keys(run, {"value", "text", "count"})
                if type(run["count"]) is not int or not 1 <= run["count"] <= 1_000_000_000:
                    fail("invalid_count")
                for name in ("value", "text"):
                    if run[name] is not None and not isinstance(run[name], str):
                        fail("schema_string")
                if run["value"] is not None:
                    try:
                        numeric = Decimal(run["value"])
                    except InvalidOperation as exc:
                        raise SummaryContractError("invalid_numeric_string") from exc
                    if not numeric.is_finite():
                        fail("invalid_numeric_string")
                pair = (run["value"], run["text"])
                if pair == previous:
                    fail("noncanonical_adjacent_runs")
                previous = pair


def ordered_source(prepared_text: str, instruction: str) -> dict[str, Any]:
    """Use exactly M1's retained text, without legacy distinct-value grouping."""
    prefix = instruction.strip()
    if prepared_text.strip() == prefix:
        return {"windows": []}
    if not prepared_text.startswith(prefix + "\n"):
        fail("source_instruction_mismatch")
    document: dict[str, Any] = {"windows": []}
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    window = None
    for raw in prepared_text[len(prefix) + 1:].splitlines():
        line = raw.strip()
        if not line:
            continue
        if TIME_LABEL_RE.fullmatch(line):
            window = {"label": line, "facts": []}
            document["windows"].append(window)
            groups = {}
            continue
        if window is None:
            fail("source_fact_without_window")
        if not line.startswith("- "):
            fail("source_parse_error")
        parts = SOURCE_FIELD_RE.split(line[2:])
        code, fields = parts[0], {}
        for part in parts[1:]:
            name, value = part.split("=", 1)
            if name in fields:
                fail("ambiguous_source_fields")
            fields[name] = value
        value, text = fields.get("value"), fields.get("text_value")
        unit, source = fields.get("unit"), fields.get("source")
        key = (code, source, unit)
        if key not in groups:
            groups[key] = {"code": code, "source": source, "unit": unit, "sequence": []}
            window["facts"].append(groups[key])
        sequence = groups[key]["sequence"]
        if sequence and (sequence[-1]["value"], sequence[-1]["text"]) == (value, text):
            sequence[-1]["count"] += 1
        else:
            sequence.append({"value": value, "text": text, "count": 1})
    check_shape(document)
    if not document["windows"]:
        fail("source_missing_windows")
    return document


def trajectory(fact: dict[str, Any]) -> dict[str, str] | None:
    """Exact, unit-specific endpoint change and monotonicity; no clinical interpretation.

    Missing numeric values make the full trajectory unknown. Text qualifiers are
    retained verbatim, so no diagnostic interpretation is inferred from them.
    """
    runs = fact["sequence"]
    if any(run["value"] is None for run in runs) or sum(run["count"] for run in runs) < 2:
        return None
    values = [Decimal(run["value"]) for run in runs]
    directions = {1 if b > a else -1 if b < a else 0 for a, b in pairwise(values)}
    movement = "fluctuating" if {-1, 1} <= directions else (
        "nondecreasing" if 1 in directions else "nonincreasing" if -1 in directions else "constant"
    )
    net = "higher" if values[-1] > values[0] else "lower" if values[-1] < values[0] else "unchanged"
    return {"net_change": net, "movement": movement}


def features(document: dict[str, Any]) -> dict[str, Counter]:
    output = {name: Counter() for name in ("concepts", "numbers", "text_values", "units", "recurrence", "trends")}
    for window in document["windows"]:
        for fact in window["facts"]:
            key = (window["label"], fact["code"], fact["source"], fact["unit"])
            output["concepts"][key] = 1
            if fact["unit"] is not None:
                output["units"][key] = 1
            for run in fact["sequence"]:
                for field, name in (("value", "numbers"), ("text", "text_values")):
                    if run[field] is not None:
                        output[name][(*key, run[field])] = 1
            count = sum(run["count"] for run in fact["sequence"])
            if count > 1:
                output["recurrence"][key] = count
            direction = trajectory(fact)
            if direction is not None:
                output["trends"][(*key, compact(direction))] = 1
    return output


def render(document: dict[str, Any]) -> str:
    """Deterministic compact display of model-returned facts, never a source fallback."""
    check_shape(document)
    if not document["windows"]:
        return "[NO_PRE_T_EVENTS]"
    lines = []
    for window in document["windows"]:
        lines.append(window["label"])
        for fact in window["facts"]:
            # Tuple-like JSON preserves escaped pipes/brackets and null vs empty values.
            runs = [[r["value"], r["text"], r["count"]] for r in fact["sequence"]]
            parts = [f'- {fact["code"]}']
            for field in ("source", "unit"):
                if fact[field] is not None:
                    parts.append(f"{field}={compact(fact[field])}")
            if len(runs) == 1:
                value, text, count = runs[0]
                if value is not None:
                    parts.append(f"value={value}")
                if text is not None:
                    parts.append(f"text_value={compact(text)}")
                if count > 1:
                    parts.append(f"occurrences={count}")
            else:
                parts.append(f"ordered_value_text_count={compact(runs)}")
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def validate(document: dict[str, Any], source: dict[str, Any], *, thresholds: dict, token_counter: Any) -> dict:
    check_shape(source)
    check_shape(document)
    if [w["label"] for w in document["windows"]] != [w["label"] for w in source["windows"]]:
        fail("time_sequence_changed")
    for candidate_window, source_window in zip(document["windows"], source["windows"], strict=True):
        remaining = iter(source_window["facts"])
        for candidate in candidate_window["facts"]:
            if not any(candidate == original for original in remaining):
                fail("invented_changed_or_reordered_series")
    before, after = features(source), features(document)
    rates = {}
    minimums = {"concepts": "concept", "numbers": "numeric", "text_values": "numeric",
                "units": "unit", "recurrence": "recurrence", "trends": "trend"}
    for name, original in before.items():
        denominator = sum(original.values())
        rates[name] = sum((original & after[name]).values()) / denominator if denominator else 1.0
        if rates[name] < thresholds[f"minimum_{minimums[name]}_retention"]:
            fail(f"retention_{name}")
    summary = render(document)
    count = token_counter(summary)
    if type(count) is not int or not 0 < count <= 8192:
        fail("embedding_token_budget")
    return {"status": "PASS", "version": VERSION, "retention": rates, "time_sequence_exact": True,
            "embedding_tokens": count, "summary_hash": stable_hash(summary),
            "source_hash": stable_hash(source), "document_hash": stable_hash(document),
            "character_compression_ratio": len(summary) / max(1, len(render(source)))}


def generation_identity(source: dict, condition: str, config: dict, served_identity: str) -> str:
    return stable_hash({"version": VERSION, "source": source, "condition": condition,
                        "instruction_model": config["instruction_model"], "decoding": config["decoding"],
                        "prompt": PROMPT, "schema": response_schema(),
                        "served_identity": served_identity})


def synthetic_source() -> dict:
    return ordered_source(
        "Synthetic instruction\n[TIME=31_60_DAYS_BEFORE_T]\n"
        "- SYN/A | value=1 | unit=u | source=measurement\n"
        "- SYN/A | value=2 | unit=u | source=measurement\n"
        "- SYN/A | value=1 | unit=u | source=measurement\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n- SYN/B | text_value=negative | source=measurement",
        "Synthetic instruction",
    )


def request_payload(source: dict, config: dict) -> dict:
    return {"model": config["instruction_model"], "messages": [
        {"role": "system", "content": PROMPT}, {"role": "user", "content": compact(source)}],
        **copy.deepcopy(config["decoding"]),
        "response_format": {"type": "json_schema", "json_schema": {
            "name": VERSION, "strict": True, "schema": response_schema()}}, "stream": False}
