"""Prompt-only generation amendment after 986; fidelity is deliberately unchanged."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_render_v2 as renderer
from temporal_ehr.final_m23.structured_summary import (
    PROMPT as ORIGINAL_PROMPT,
)
from temporal_ehr.final_m23.structured_summary import (
    SummaryContractError,
    request_payload,
    synthetic_source,
)
from temporal_ehr.serialization.integrity import stable_hash

VERSION = "m2_ordered_structured_v2"
PROMPT = ORIGINAL_PROMPT + """

Copying rules (mandatory):
1. Traverse windows and facts in their supplied array order. Each retained fact is one
   indivisible object. Emit it once, at its original relative position. Never revisit it.
2. The identity within a window is the full (code, source, unit) tuple, not code alone.
   The same code with a DIFFERENT source or unit is a DIFFERENT fact; do not combine them.
3. Copy the ENTIRE sequence array of a retained fact. Preserve every run, its position,
   numeric string spelling, text string, null, and integer count. Count is supplied data,
   not a calculation. Never recount, sum, interpolate, sort, or deduplicate runs.
4. A return to an earlier value is intentional. For example, runs with values "1", "2",
   "1" are three runs in that order, not "1", "2" and not a single averaged value.
   A count of 3 on the first run stays 3; a later count of 1 stays 1.
5. Null is not zero, an empty string, or the string "null". Preserve these distinctions.
6. If omitting a whole fact, move forward to the next source fact. Do not replace it with
   a paraphrase or partially copy its sequence. All original retention requirements apply.
Before ending the JSON, check that no retained tuple was emitted twice in the same window,
every retained sequence is unchanged, and the retained facts form a source-order subsequence.
Do not output this checklist, explanations, or additional fields. Produce the summary once.
"""


def payload(source: dict, config: dict) -> dict:
    result = request_payload(source, config)
    result["messages"][0]["content"] = PROMPT
    # Stable object-key serialization across atomic JSON save/reload; arrays NEVER sort.
    result["messages"][1]["content"] = json.dumps(
        source, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    result["response_format"]["json_schema"]["name"] = VERSION
    return result


def identity(item: dict, config: dict, served: str) -> str:
    return stable_hash({"version": VERSION, "request": payload(item["source"], config),
                        "condition": item["condition"], "served_identity": served,
                        "attempt_policy": "two_fixed_requests_first_valid_no_feedback"})


def probes() -> list[dict]:
    first = synthetic_source()
    fact = first["windows"][0]["facts"][0]
    fact["sequence"][0]["count"] = 3
    other = copy.deepcopy(fact)
    other["unit"] = "different-unit"
    first["windows"][0]["facts"].append(other)
    second = copy.deepcopy(first)
    second["windows"].reverse()  # Supplied falsified labels must not be repaired.
    second["windows"][0]["facts"][0]["sequence"] = [
        {"value": None, "text": "", "count": 1},
        {"value": "0", "text": "null", "count": 2},
        {"value": None, "text": None, "count": 1},
    ]
    return [first, second, {"windows": []}]


def assess(item: dict, config: dict, thresholds: dict, counter: Any, directory: Path,
           served: str, tokenizer_hash: str, client: Any = None) -> dict:
    """Two same-contract attempts; never repair, merge or fill an LLM response."""
    directory.mkdir(parents=True, exist_ok=False)
    old.write_once(directory / "source.json", item)
    request = payload(item["source"], config)
    old.write_once(directory / "request.json", request)
    client = client or old.PrivateClient(config)
    errors, rejected = [], set()
    for attempt in (1, 2):
        try:
            response = client.request("chat/completions", request)
            old.write_once(directory / f"response-{attempt}.json", response)
            document = old.response_document(response)
            digest = stable_hash(document)
            if digest in rejected:
                raise SummaryContractError("identical_rejected_content")
            rejected.add(digest)
            validation = renderer.validate(document, item["source"], thresholds=thresholds,
                                           token_counter=counter)
            text = renderer.render(document)
            key = identity(item, config, served)
            result = {"status": "PASS", "document": document, "summary": text,
                      "validation": validation, "attempts": attempt, "errors": errors,
                      "generation_key": key, "embedding_input_key": renderer.embedding_identity(
                          key, text, tokenizer_hash)}
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


def verify_item(directory: Path, item: dict, config: dict, thresholds: dict, counter: Any,
                served: str, tokenizer_hash: str) -> dict:
    result = old.read(directory / "assessment.json")
    attempt = result.get("attempts")
    if (result.get("status") != "PASS" or type(attempt) is not int or attempt not in (1, 2)
            or old.read(directory / "source.json") != item
            or old.read(directory / "request.json") != payload(item["source"], config)):
        raise old.ContractError("Invalid v2 source/request/admission binding.")
    document = old.response_document(old.read(directory / f"response-{attempt}.json"))
    validation = renderer.validate(document, item["source"], thresholds=thresholds, token_counter=counter)
    key = identity(item, config, served)
    text = renderer.render(document)
    if (document != result["document"] or validation != result["validation"] or text != result["summary"]
            or result["generation_key"] != key or result["embedding_input_key"] !=
            renderer.embedding_identity(key, text, tokenizer_hash)):
        raise old.ContractError("V2 response or derived input failed revalidation.")
    # A later response cannot replace an earlier valid response (no cherry-picking retries).
    for previous in range(1, attempt):
        path = directory / f"response-{previous}.json"
        if not path.exists():
            continue  # A recorded transport failure may have no response body.
        try:
            candidate = old.response_document(old.read(path))
            renderer.validate(candidate, item["source"], thresholds=thresholds, token_counter=counter)
        except (SummaryContractError, old.NonRetryableSummaryError):
            continue
        raise old.ContractError("Earlier valid response was skipped.")
    return result
