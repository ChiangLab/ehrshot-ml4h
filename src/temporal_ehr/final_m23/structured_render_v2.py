"""Reversible compact rendering of Qwen-returned structured facts; no source fallback."""

from __future__ import annotations

import re
from typing import Any

from temporal_ehr.final_m23.structured_summary import (
    TIME_LABEL_RE,
    SummaryContractError,
    check_shape,
    compact,
    strict_json,
)
from temporal_ehr.final_m23.structured_summary import (
    validate as validate_v1,
)
from temporal_ehr.serialization.integrity import stable_hash

VERSION = "m2_ordered_render_v2"
HEADER = "Fields: s=source;u=unit;v=value;x=text;n=count;r=ordered[value,text,count]. Default count=1."
BARE = re.compile(r"[A-Za-z0-9_./%:+-]+")


def encode_string(value: str) -> str:
    return value if BARE.fullmatch(value) else compact(value)


def decode_string(value: str) -> str:
    if BARE.fullmatch(value):
        return value
    decoded = strict_json(value)
    if not isinstance(decoded, str):
        raise SummaryContractError("render_v2_string")
    return decoded


def render(document: dict) -> str:
    check_shape(document)
    lines = [HEADER]
    if not document["windows"]:
        return HEADER + "\n[NO_PRE_T_EVENTS]"
    for window in document["windows"]:
        lines.append(window["label"])
        for fact in window["facts"]:
            parts = [encode_string(fact["code"])]
            for field, short in (("source", "s"), ("unit", "u")):
                if fact[field] is not None:
                    parts.append(short + "=" + encode_string(fact[field]))
            runs = fact["sequence"]
            if len(runs) == 1:
                run = runs[0]
                for field, short in (("value", "v"), ("text", "x")):
                    if run[field] is not None:
                        parts.append(short + "=" + encode_string(run[field]))
                if run["count"] != 1:
                    parts.append("n=" + str(run["count"]))
            else:
                parts.append("r=" + compact([[r["value"], r["text"], r["count"]] for r in runs]))
            lines.append("\t".join(parts))
    return "\n".join(lines)


def decode(text: str) -> dict:
    """Read exactly the canonical format; round trip proves no fact/order/count loss."""
    lines = text.split("\n")
    if not lines or lines[0] != HEADER:
        raise SummaryContractError("render_v2_header")
    document: dict[str, Any] = {"windows": []}
    if lines[1:] == ["[NO_PRE_T_EVENTS]"]:
        return document
    window = None
    for line in lines[1:]:
        if TIME_LABEL_RE.fullmatch(line):
            window = {"label": line, "facts": []}
            document["windows"].append(window)
            continue
        if window is None:
            raise SummaryContractError("render_v2_missing_window")
        parts = line.split("\t")
        fields = {}
        for part in parts[1:]:
            name, separator, value = part.partition("=")
            if not separator or name not in {"s", "u", "v", "x", "n", "r"} or name in fields:
                raise SummaryContractError("render_v2_fields")
            fields[name] = value
        if "r" in fields:
            if set(fields) & {"v", "x", "n"}:
                raise SummaryContractError("render_v2_ambiguous_runs")
            tuples = strict_json(fields["r"])
            if not isinstance(tuples, list) or any(not isinstance(r, list) or len(r) != 3 for r in tuples):
                raise SummaryContractError("render_v2_runs")
            runs = [{"value": v, "text": t, "count": n} for v, t, n in tuples]
        else:
            count = fields.get("n", "1")
            if not re.fullmatch(r"[1-9][0-9]*", count):
                raise SummaryContractError("render_v2_count")
            runs = [{"value": decode_string(fields["v"]) if "v" in fields else None,
                     "text": decode_string(fields["x"]) if "x" in fields else None,
                     "count": int(count)}]
        window["facts"].append({"code": decode_string(parts[0]),
                                "source": decode_string(fields["s"]) if "s" in fields else None,
                                "unit": decode_string(fields["u"]) if "u" in fields else None,
                                "sequence": runs})
    check_shape(document)
    if render(document) != text:
        raise SummaryContractError("render_v2_noncanonical")
    return document


def validate(document: dict, source: dict, *, thresholds: dict, token_counter: Any) -> dict:
    # Reuse every v1 factual gate. Its obsolete rendering's size is irrelevant here;
    # the actual new text is counted below, without truncation or a larger budget.
    result = validate_v1(document, source, thresholds=thresholds, token_counter=lambda _: 1)
    summary = render(document)
    if decode(summary) != document:
        raise SummaryContractError("render_v2_roundtrip")
    count = token_counter(summary)
    if type(count) is not int or not 0 < count <= 8192:
        raise SummaryContractError("embedding_token_budget")
    return {**result, "renderer_version": VERSION, "roundtrip_exact": True,
            "embedding_tokens": count, "summary_hash": stable_hash(summary),
            "character_compression_ratio": len(summary) / max(1, len(render(source)))}


def embedding_identity(generation_key: str, summary: str, tokenizer_hash: str) -> str:
    return stable_hash({"generation_key": generation_key, "renderer_version": VERSION,
                        "summary": summary, "tokenizer_hash": tokenizer_hash,
                        "encoder": "Qwen/Qwen3-Embedding-0.6B", "maximum_tokens": 8192})
