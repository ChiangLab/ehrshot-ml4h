"""On-premise one-pass summaries, immutable caches, and fidelity validation."""

from __future__ import annotations

import ipaddress
import json
import re
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import load_final_m1_authorization
from temporal_ehr.final_m1.embedding import condition_label, condition_slug
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.final_m23.protocol import FinalM23Authorization, load_final_m23_authorization
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, atomic_json, file_sha256
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash

TIME_LABEL_RE = re.compile(r"\[TIME=\d+_\d+_DAYS_BEFORE_T\]")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
UNIT_RE = re.compile(r"\bunit=([^|\n]+)")
VALUE_RE = re.compile(r"\bvalue=([-+]?\d+(?:\.\d+)?)")
VALUES_RE = re.compile(r"\bvalues=([^|\n]+)")
OCCURRENCES_RE = re.compile(r"\boccurrences=(\d+)")
NUMBER_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
TREND_WORDS = {"increasing", "decreasing", "stable"}
NULL_LIKE_SUMMARIES = frozenset({"none", "null"})
M2_SOURCE_COMPACTION_SCHEMA = "m2_grouped_facts_v2"
M2_SUMMARY_CACHE_SCHEMA = "m2_summary_cache_v4_grouped_completion_bound"
M2_EMBEDDING_MAX_TOKENS = 8192
SOURCE_FIELD_RE = re.compile(r" \| (?=(?:value|text_value|unit|source)=)")
MANUAL_REVIEW_CRITERIA = (
    "concepts_preserved",
    "numeric_values_preserved",
    "units_preserved",
    "recurrence_and_trends_preserved",
    "time_labels_preserved",
    "no_unsupported_concepts",
    "no_contradictions",
    "no_timing_repair",
)


@dataclass(frozen=True)
class SummaryGeneration:
    """Validated assistant text plus non-clinical completion provenance."""

    summary: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None


class NonRetryableSummaryError(RuntimeError):
    """A deterministic response-contract failure that retries cannot repair."""


def validate_on_premise_api_base(api_base: str) -> str:
    """Reject public, credential-bearing, TLS-proxied, or non-HTTP summary endpoints."""
    parsed = urlparse(api_base)
    if parsed.scheme != "http" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ContractError("M2 summary API must be a plain credential-free on-premise HTTP URL.")
    try:
        address = ipaddress.ip_address(parsed.hostname or "")
    except ValueError as exc:
        raise ContractError("M2 summary API must use a literal private IP address.") from exc
    if not address.is_private or address.is_loopback or parsed.port is None:
        raise ContractError("M2 summary API must use a non-loopback RFC1918 address and explicit port.")
    return api_base.rstrip("/") + "/"


class OnPremiseSummaryClient:
    """Minimal OpenAI-compatible client that cannot route governed text externally."""

    def __init__(self, *, api_base: str, model: str, timeout_seconds: int = 600) -> None:
        self.api_base = validate_on_premise_api_base(api_base)
        self.model = model
        self.timeout_seconds = int(timeout_seconds)

    def _request(self, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
        request = urllib.request.Request(
            urljoin(self.api_base, path.lstrip("/")),
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError("Authorized on-premise summary endpoint request failed.") from exc
        if not isinstance(result, dict):
            raise RuntimeError("Authorized on-premise summary endpoint returned a non-object response.")
        return result

    def model_revision_hash(self) -> str:
        payload = self._request("models")
        models = payload.get("data", [])
        record = next((item for item in models if isinstance(item, dict) and item.get("id") == self.model), None)
        if record is None:
            raise ContractError("The locked M2 instruction model is not served by the on-premise endpoint.")
        # `created` is a vLLM process timestamp, not a model-weight revision. Including it caused
        # identical named deployments to invalidate resumable caches after every service restart.
        stable_identity = {
            "schema": "served_model_identity_v2",
            **{
                field: record.get(field)
                for field in ("id", "root", "parent", "owned_by", "max_model_len")
            },
        }
        return stable_hash(stable_identity)

    def summarize_response(
        self,
        *,
        system_prompt: str,
        source_text: str,
        decoding: Mapping[str, Any],
    ) -> SummaryGeneration:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
            "temperature": float(decoding["temperature"]),
            "top_p": float(decoding["top_p"]),
            "seed": int(decoding["seed"]),
            "max_tokens": int(decoding["max_output_tokens"]),
            "chat_template_kwargs": dict(decoding["chat_template_kwargs"]),
            "stream": False,
        }
        result = self._request("chat/completions", payload)
        try:
            choice = result["choices"][0]
            content = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
        except (KeyError, IndexError, TypeError) as exc:
            raise NonRetryableSummaryError(
                "On-premise summary response lacks assistant content or finish metadata."
            ) from exc
        if not isinstance(content, str):
            raise NonRetryableSummaryError("On-premise summary response content was not text.")
        summary = content.strip()
        if not summary or summary.casefold() in NULL_LIKE_SUMMARIES:
            raise NonRetryableSummaryError(
                "On-premise summary response was empty or null-like."
            )
        if finish_reason != "stop":
            raise NonRetryableSummaryError(
                f"On-premise summary did not finish cleanly (finish_reason={finish_reason!r})."
            )
        usage = result.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return SummaryGeneration(
            summary=summary,
            finish_reason=finish_reason,
            prompt_tokens=int(prompt_tokens) if isinstance(prompt_tokens, int) else None,
            completion_tokens=int(completion_tokens) if isinstance(completion_tokens, int) else None,
        )

    def summarize(
        self,
        *,
        system_prompt: str,
        source_text: str,
        decoding: Mapping[str, Any],
    ) -> str:
        """Compatibility wrapper for synthetic probes that need text only."""
        return self.summarize_response(
            system_prompt=system_prompt,
            source_text=source_text,
            decoding=decoding,
        ).summary


def summary_cache_key(
    *,
    source_hash: str,
    condition: str,
    prompt_hash: str,
    model_revision_hash: str,
    decoding_hash: str,
) -> str:
    """Hash every required M2 generation input; patient identity is not needed."""
    return stable_hash(
        {
            "schema": M2_SUMMARY_CACHE_SCHEMA,
            "source_hash": source_hash,
            "condition": condition,
            "prompt_hash": prompt_hash,
            "model_revision_hash": model_revision_hash,
            "decoding_hash": decoding_hash,
        }
    )


def _summarize_with_retries(
    client: OnPremiseSummaryClient,
    *,
    system_prompt: str,
    source_text: str,
    decoding: Mapping[str, Any],
    retries: int,
) -> SummaryGeneration:
    """Retry transient private-endpoint failures without changing generation inputs."""
    for attempt in range(retries + 1):
        try:
            return client.summarize_response(
                system_prompt=system_prompt,
                source_text=source_text,
                decoding=decoding,
            )
        except NonRetryableSummaryError:
            raise
        except RuntimeError:
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def compact_exact_duplicate_facts(source_text: str) -> str:
    """Collapse exact repeated fact lines within each window using deterministic counts.

    The serialized history does not expose event timestamps within a window. Removing repeated
    copies of the exact same rendered fact therefore loses no available order information, while
    the explicit count retains recurrence and avoids asking the instruction model to count a long
    run of identical tokens.
    """
    output: list[str] = []
    facts: list[str] = []

    def flush_facts() -> None:
        counts = Counter(facts)
        emitted: set[str] = set()
        for fact in facts:
            if fact in emitted:
                continue
            emitted.add(fact)
            count = counts[fact]
            if count > 1:
                if OCCURRENCES_RE.search(fact):
                    raise ContractError("M2 source fact already contains a recurrence count.")
                output.append(f"{fact} | occurrences={count}")
            else:
                output.append(fact)
        facts.clear()

    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            facts.append(line)
            continue
        flush_facts()
        output.append(line)
    flush_facts()
    compacted = "\n".join(output)
    if not compacted:
        raise ContractError("M2 deterministic source compaction produced empty text.")
    return compacted


def _parse_rendered_source_fact(
    line: str,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    """Parse one locked serializer fact without interpreting its clinical content."""
    if not line.startswith("- ") or OCCURRENCES_RE.search(line):
        raise ContractError("M2 grouping requires an ungrouped locked serializer fact line.")
    parts = SOURCE_FIELD_RE.split(line[2:])
    concept = parts[0].strip()
    if not concept:
        raise ContractError("M2 source fact has no clinical code field.")
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ContractError("M2 source fact contains an unrecognized structured field.")
        key, value = part.split("=", maxsplit=1)
        if key not in {"value", "text_value", "unit", "source"} or key in fields:
            raise ContractError("M2 source fact contains duplicate or unsupported fields.")
        fields[key] = value
    return concept, fields.get("value"), fields.get("text_value"), fields.get("unit"), fields.get("source")


def group_structured_facts(source_text: str) -> str:
    """Losslessly group rendered facts by concept/source/unit inside unchanged windows.

    Values and text values retain first-occurrence order, recurrence is counted in Python, and
    neither time-block order nor first concept occurrence is changed. This makes the instruction
    model copy a bounded structured representation instead of performing unreliable aggregation.
    """
    output: list[str] = []
    window_facts: list[tuple[str, str | None, str | None, str | None, str | None]] = []

    def flush_window() -> None:
        grouped: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
        order: list[tuple[str, str | None, str | None]] = []
        for concept, value, text_value, unit, source in window_facts:
            key = (concept, source, unit)
            if key not in grouped:
                grouped[key] = {"values": [], "text_values": [], "occurrences": 0}
                order.append(key)
            record = grouped[key]
            record["occurrences"] += 1
            if value is not None and value not in record["values"]:
                record["values"].append(value)
            if text_value is not None and text_value not in record["text_values"]:
                record["text_values"].append(text_value)
        for concept, source, unit in order:
            record = grouped[(concept, source, unit)]
            parts = [f"- {concept}"]
            if record["values"]:
                parts.append("values=" + ",".join(record["values"]))
            if record["text_values"]:
                encoded = json.dumps(record["text_values"], ensure_ascii=False, separators=(",", ":"))
                parts.append(f"text_values={encoded}")
            if unit is not None:
                parts.append(f"unit={unit}")
            if source is not None:
                parts.append(f"source={source}")
            parts.append(f"occurrences={record['occurrences']}")
            output.append(" | ".join(parts))
        window_facts.clear()

    for raw_line in source_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            window_facts.append(_parse_rendered_source_fact(line))
            continue
        flush_window()
        if not TIME_LABEL_RE.fullmatch(line):
            raise ContractError("M2 source contains a line outside the locked window/fact grammar.")
        output.append(line)
    flush_window()
    grouped_text = "\n".join(output)
    if not grouped_text:
        raise ContractError("M2 deterministic source grouping produced empty text.")
    return grouped_text


class LocalSummaryTokenValidator:
    """Offline tokenizer gate for the frozen downstream Qwen embedding context."""

    def __init__(self, model_directory: Path, *, maximum_tokens: int) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - DGX dependency gate
            raise RuntimeError("M2 summary validation requires transformers in the container.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_directory),
            local_files_only=True,
        )
        self.maximum_tokens = int(maximum_tokens)

    def __call__(self, summary: str) -> int:
        token_count = len(self.tokenizer.encode(summary, add_special_tokens=False))
        if token_count > self.maximum_tokens:
            raise NonRetryableSummaryError(
                "M2 summary exceeds the frozen downstream 8,192-token embedding context."
            )
        return token_count


def m2_source_text(prepared_text: str, instruction: str) -> str:
    """Remove M1's prediction instruction and retain only ordered observation windows."""
    prefix = instruction.strip()
    expected = f"{prefix}\n"
    if prepared_text.strip() == prefix:
        return "[NO_PRE_T_EVENTS]"
    if not prepared_text.startswith(expected):
        raise ContractError("Prepared M1 text does not begin with its locked task instruction.")
    source = prepared_text[len(expected) :].strip()
    if not source or "[" not in source:
        raise ContractError("M2 source contains no serialized observation window.")
    return group_structured_facts(source)


class SummaryCache:
    """One immutable private JSON record per summary with safe resume semantics."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, cache_key: str) -> Path:
        return self.root / "rows" / f"{cache_key}.json"

    def load(self, cache_key: str, expected: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.path(cache_key)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(payload.get(field) != value for field, value in expected.items()):
            raise ContractError("Existing M2 summary cache row does not match its immutable inputs.")
        summary = payload.get("summary")
        if (
            not isinstance(summary, str)
            or not summary.strip()
            or summary.strip().casefold() in NULL_LIKE_SUMMARIES
            or payload.get("summary_hash") != stable_hash(summary)
            or payload.get("finish_reason") != "stop"
        ):
            raise ContractError(
                "Existing M2 summary cache row is empty, truncated, or hash-mismatched."
            )
        return payload

    def store(self, cache_key: str, payload: Mapping[str, Any]) -> Path:
        path = self.path(cache_key)
        existing = self.load(cache_key, {key: value for key, value in payload.items() if key != "summary"})
        if existing is not None:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, payload)
        return path


def m2_embedding_root(
    spec: FinalM23Authorization,
    *,
    task: str,
    label: str,
    source_git_commit: str,
) -> Path:
    """Isolate source-bound embedding attempts while allowing summary-row reuse."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_git_commit):
        raise ContractError("M2 embedding namespace requires a full source Git commit.")
    return (
        spec.private_root
        / "m2"
        / "embeddings"
        / f"source_{source_git_commit}"
        / task
        / condition_slug(label)
    )


def _condition_rows(rows: Sequence[Mapping[str, Any]], label: str) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if condition_label(row) == label]
    selected.sort(key=lambda row: (str(row["split"]), str(row["instance_key"])))
    return selected


def m2_stage_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    spec: FinalM23Authorization,
    stage: str,
) -> list[dict[str, Any]]:
    """Return reference/fidelity rows first, or selected-approach sensitivities later."""
    if any(row.get("split") not in {"train", "validation"} for row in rows):
        raise ContractError("M2 received a row outside train/validation.")
    if stage == "reference":
        c1 = _condition_rows(rows, str(spec.m2["reference_condition"]))
        c3_label = f"{spec.m2['fidelity_condition']}:seed={spec.m2['fidelity_seed']}"
        c3_validation = [row for row in _condition_rows(rows, c3_label) if row["split"] == "validation"]
        sample_size = int(spec.m2["fidelity"]["histories_per_task_condition"])
        c3_validation.sort(key=lambda row: stable_hash((row["task"], c3_label, row["instance_key"])))
        if len(c3_validation) < sample_size:
            raise ContractError("M2 lacks enough shuffled validation histories for fidelity sampling.")
        return [*c1, *c3_validation[:sample_size]]
    if stage != "sensitivity":
        raise ValueError("M2 stage must be reference or sensitivity.")
    selection_path = spec.aggregate_root / "approach_selection.json"
    if not selection_path.is_file():
        raise ContractError("M2 sensitivity requires the completed reference approach selection.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selected_approach") != "M2" or selection.get("reference_cells_passed") is not True:
        raise ContractError("M2 sensitivity is not authorized by the frozen reference selection.")
    output: list[dict[str, Any]] = []
    for label in spec.m2["sensitivity_conditions"]:
        output.extend(_condition_rows(rows, str(label)))
    return output


def _write_parquet_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        raise ContractError("Refusing to write empty Prompt 4 private metadata.")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if not pq.read_table(path).equals(table):
            raise ContractError("Existing Prompt 4 private metadata drifted.")
    else:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return file_sha256(path)


def _embed_summaries(
    *,
    spec: FinalM23Authorization,
    task: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    encoder: LocalQwenEncoder,
    identity_hash: str,
    tokenizer_hash: str,
    source_git_commit: str,
) -> dict[str, Any]:
    selected = [row for row in rows if condition_label(row) == label]
    if not selected:
        raise ContractError(f"M2 has no rows to embed for {task}/{label}.")
    summary_keys = [str(summaries[str(row["representation_key"])]["cache_key"]) for row in selected]
    row_keys = [
        stable_hash({"summary_cache_key": summary_key, "instance_key": row["instance_key"]})
        for row, summary_key in zip(selected, summary_keys, strict=True)
    ]
    root = m2_embedding_root(
        spec,
        task=task,
        label=label,
        source_git_commit=source_git_commit,
    )
    configuration_hash = stable_hash(
        {
            "experiment": spec.experiment_id,
            "config": stable_hash(spec.raw),
            "source_git_commit": source_git_commit,
            "task": task,
            "condition": label,
            "summary_cache_keys": summary_keys,
            "embedding_row_keys": row_keys,
            "model_artifact_hash": identity_hash,
            "tokenizer_hash": tokenizer_hash,
        }
    )
    metadata = [
        {
            "experiment_id": spec.experiment_id,
            "task": task,
            "split": row["split"],
            "instance_key": row["instance_key"],
            "target": row["target"],
            "condition": label,
            "summary_cache_key": summary_key,
            "embedding_cache_key": key,
            "embedding_row_index": index,
            "condition_embedding_row_index": index,
            "mirrors_embedding_row": True,
        }
        for index, (row, key, summary_key) in enumerate(
            zip(selected, row_keys, summary_keys, strict=True)
        )
    ]
    metadata_hash = _write_parquet_once(root / "patient_metadata.parquet", metadata)
    cache = ResumableEmbeddingCache(
        root,
        row_keys=row_keys,
        dimension=encoder.dimension,
        configuration_hash=configuration_hash,
    )
    pending = cache.pending_indices()
    if not cache.complete_path.is_file():
        cache.begin_attempt()
        for offset in range(0, len(pending), spec.model_batch_size):
            indices = pending[offset : offset + spec.model_batch_size]
            texts = [
                str(summaries[str(selected[int(index)]["representation_key"])]["summary"])
                for index in indices
            ]
            cache.write_rows(indices.tolist(), encoder.encode(texts, batch_size=spec.model_batch_size))
        cache.finalize()
        cache.finish_attempt()
    vectors = cache.load_complete()
    norms = np.linalg.norm(np.asarray(vectors, dtype=np.float64), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0.0):
        raise ContractError("M2 summary embeddings are not FP32 L2-normalized.")
    return {
        "condition": label,
        "rows": len(selected),
        "dimension": encoder.dimension,
        "configuration_hash": configuration_hash,
        "metadata_sha256": metadata_hash,
        "cache_complete_sha256": file_sha256(cache.complete_path),
        "rows_computed_this_attempt": len(pending),
        "cache_reused_without_encoding": len(pending) == 0,
    }


def run_m2_task(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    task: str,
    stage: str,
    source_git_commit: str,
    client: OnPremiseSummaryClient | None = None,
    embed_summaries: bool = True,
    summary_token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """Generate/cache one-pass summaries and embed the stage-authorized conditions."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    if task not in spec.tasks or len(source_git_commit) != 40:
        raise ContractError("M2 task or source commit is invalid.")
    m1 = load_final_m1_authorization(spec.m1_config_path, repository_root=repository_root)
    rows = m2_stage_rows(read_final_task_texts(m1, task), spec=spec, stage=stage)
    if not rows:
        raise ContractError("M2 stage selected no prepared histories.")
    summary_client = client or OnPremiseSummaryClient(
        api_base=str(spec.m2["api_base"]),
        model=str(spec.m2["instruction_model"]),
    )
    model_revision_hash = summary_client.model_revision_hash()
    prompt = str(spec.m2["prompt"])
    prompt_hash = stable_hash({"version": spec.m2["prompt_version"], "prompt": prompt})
    decoding = dict(spec.m2["decoding"])
    decoding_hash = stable_hash(decoding)
    concurrent_requests = int(spec.m2["execution"]["concurrent_requests"])
    request_retries = int(spec.m2["execution"]["request_retries"])
    token_counter = summary_token_counter or LocalSummaryTokenValidator(
        model_root / spec.model_directory_name,
        maximum_tokens=M2_EMBEDDING_MAX_TOKENS,
    )
    cache = SummaryCache(spec.private_root / "m2" / "summaries" / task)
    representation_keys: dict[str, str] = {}
    payloads_by_key: dict[str, dict[str, Any]] = {}
    pending_by_key: dict[str, tuple[str, dict[str, Any]]] = {}
    reused = 0
    started = time.perf_counter()
    for row in rows:
        label = condition_label(row)
        source_text = m2_source_text(str(row["text"]), m1.instructions[task])
        source_hash = stable_hash(
            {"schema": M2_SOURCE_COMPACTION_SCHEMA, "source_text": source_text}
        )
        key = summary_cache_key(
            source_hash=source_hash,
            condition=label,
            prompt_hash=prompt_hash,
            model_revision_hash=model_revision_hash,
            decoding_hash=decoding_hash,
        )
        expected = {
            "cache_key": key,
            "source_hash": source_hash,
            "condition": label,
            "prompt_hash": prompt_hash,
            "model_revision_hash": model_revision_hash,
            "decoding_hash": decoding_hash,
            "source_compaction_schema": M2_SOURCE_COMPACTION_SCHEMA,
        }
        if key not in payloads_by_key and key not in pending_by_key:
            payload = cache.load(key, expected)
            if payload is None:
                pending_by_key[key] = (source_text, expected)
            else:
                token_counter(str(payload["summary"]))
                payloads_by_key[key] = payload
                reused += 1
        representation_keys[str(row["representation_key"])] = key
    if pending_by_key:
        failures: list[BaseException] = []

        def generate(cache_key: str) -> tuple[str, dict[str, Any]]:
            source_text, expected = pending_by_key[cache_key]
            generation = _summarize_with_retries(
                summary_client,
                system_prompt=prompt,
                source_text=source_text,
                decoding=decoding,
                retries=request_retries,
            )
            summary_token_count = token_counter(generation.summary)
            return cache_key, {
                **expected,
                "summary": generation.summary,
                "summary_hash": stable_hash(generation.summary),
                "finish_reason": generation.finish_reason,
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "embedding_token_count": summary_token_count,
                "embedding_max_tokens": M2_EMBEDDING_MAX_TOKENS,
            }

        with ThreadPoolExecutor(max_workers=min(concurrent_requests, len(pending_by_key))) as executor:
            futures = {executor.submit(generate, key): key for key in pending_by_key}
            completed_new = 0
            for future in as_completed(futures):
                try:
                    key, payload = future.result()
                    cache.store(key, payload)
                    payloads_by_key[key] = payload
                    completed_new += 1
                    if completed_new % 25 == 0 or completed_new == len(pending_by_key):
                        print(
                            json.dumps(
                                {
                                    "event": "m2_summary_progress",
                                    "task": task,
                                    "stage": stage,
                                    "completed_new_cache_rows": completed_new,
                                    "pending_new_cache_rows": len(pending_by_key),
                                    "reused_cache_rows": reused,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except BaseException as exc:  # preserve every successful row before failing closed
                    failures.append(exc)
        if failures:
            raise RuntimeError(
                f"{len(failures)} on-premise summary request(s) failed after locked retries."
            ) from failures[0]
    summaries = {
        representation_key: payloads_by_key[cache_key]
        for representation_key, cache_key in representation_keys.items()
    }
    identity = None
    embedding_reports: list[dict[str, Any]] = []
    if embed_summaries:
        identity = load_model_identity(
            model_root / spec.model_directory_name,
            expected_repository=spec.model_repository,
        )
        encoder = LocalQwenEncoder(identity, max_seq_length=8192)
        labels = sorted(
            {
                condition_label(row)
                for row in rows
                if not (
                    stage == "reference"
                    and row["split"] == "validation"
                    and condition_label(row).startswith("C3_")
                )
            }
        )
        embedding_reports = [
            _embed_summaries(
                spec=spec,
                task=task,
                label=label,
                rows=rows,
                summaries=summaries,
                encoder=encoder,
                identity_hash=identity.artifact_hash,
                tokenizer_hash=identity.tokenizer_hash,
                source_git_commit=source_git_commit,
            )
            for label in labels
        ]
    report = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "stage": stage,
        "task": task,
        "summaries": len(rows),
        "summary_cache_rows_reused": reused,
        "summary_rows_generated_this_attempt": len(pending_by_key),
        "source_history_policy": str(spec.m2["source_history_policy"]),
        "source_compaction_schema": M2_SOURCE_COMPACTION_SCHEMA,
        "summary_cache_schema": M2_SUMMARY_CACHE_SCHEMA,
        "embedding_summary_max_tokens": M2_EMBEDDING_MAX_TOKENS,
        "maximum_embedding_summary_tokens": max(
            int(payload["embedding_token_count"]) for payload in payloads_by_key.values()
        ),
        "concurrent_summary_requests": concurrent_requests,
        "request_retries": request_retries,
        "summary_cache_key_fields": [
            "source_hash",
            "condition",
            "prompt_hash",
            "model_revision_hash",
            "decoding_hash",
        ],
        "single_pass_nonrecursive": True,
        "instruction_model": str(spec.m2["instruction_model"]),
        "instruction_model_revision_hash": model_revision_hash,
        "instruction_model_revision_kind": (
            "stable_served_endpoint_identity_excluding_process_created_timestamp"
        ),
        "on_premise_only": True,
        "external_api_used": False,
        "deterministic_decoding": True,
        "thinking_disabled": decoding["chat_template_kwargs"]["enable_thinking"] is False,
        "summary_stage_complete": True,
        "embedding_stage_complete": embed_summaries,
        "embedding_model": identity.repository if identity is not None else None,
        "embedding_model_artifact_hash": identity.artifact_hash if identity is not None else None,
        "embedding_reports": embedding_reports,
        "runtime_seconds": time.perf_counter() - started,
        "test_split_used": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    suffix = stage if embed_summaries else f"{stage}_summaries"
    report_path = spec.aggregate_root / "m2" / "tasks" / f"{task}_{suffix}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous != report:
            raise ContractError("Refusing to overwrite a different M2 task aggregate.")
    else:
        report_path.write_text(text, encoding="utf-8")
    return report


def _concepts_in_line(line: str) -> list[str]:
    bracketed = [value for value in BRACKET_RE.findall(line) if not value.startswith("TIME=")]
    if bracketed:
        return list(dict.fromkeys(bracketed))
    first_field = line.lstrip("-* ").split("|", maxsplit=1)[0].strip()
    return [first_field] if "/" in first_field else []


def _values_in_line(line: str) -> list[str]:
    values = VALUE_RE.findall(line)
    for field in VALUES_RE.findall(line):
        values.extend(NUMBER_TOKEN_RE.findall(field))
    return list(dict.fromkeys(values))


def _feature_sets(text: str) -> dict[str, Any]:
    """Extract structured facts without treating code/time digits as clinical values."""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("-")]
    concept_occurrences: Counter[str] = Counter()
    concepts: set[str] = set()
    numbers: set[str] = set()
    units: set[str] = set()
    trends: set[str] = set()
    values_by_concept: defaultdict[str, list[float]] = defaultdict(list)
    units_by_concept: defaultdict[str, set[str]] = defaultdict(set)
    facts_by_concept: defaultdict[str, set[tuple[str | None, str | None]]] = defaultdict(set)
    for line in lines:
        codes = _concepts_in_line(line)
        values = _values_in_line(line)
        unit_match = UNIT_RE.search(line)
        unit = unit_match.group(1).strip() if unit_match else None
        occurrence_match = OCCURRENCES_RE.search(line)
        occurrences = int(occurrence_match.group(1)) if occurrence_match else 1
        concepts.update(codes)
        numbers.update(values)
        if unit:
            units.add(unit)
        for code in codes:
            concept_occurrences[code] += occurrences
            if unit:
                units_by_concept[code].add(unit)
            if values:
                for value in values:
                    facts_by_concept[code].add((value, unit))
                    values_by_concept[code].append(float(value))
            else:
                facts_by_concept[code].add((None, unit))
        lowered = line.casefold()
        trends.update(word for word in TREND_WORDS if word in lowered)
    inferred_trends: set[str] = set()
    for values in values_by_concept.values():
        if len(values) >= 2:
            if values[-1] > values[0]:
                inferred_trends.add("increasing")
            elif values[-1] < values[0]:
                inferred_trends.add("decreasing")
            else:
                inferred_trends.add("stable")
    format_violations = sum(
        bool(line.strip())
        and line.strip() != "[NO_PRE_T_EVENTS]"
        and TIME_LABEL_RE.fullmatch(line.strip()) is None
        and not line.strip().startswith("-")
        for line in text.splitlines()
    )
    return {
        "concepts": Counter({concept: 1 for concept in concepts}),
        "numbers": Counter({number: 1 for number in numbers}),
        "units": Counter({unit: 1 for unit in units}),
        "time_labels": Counter(TIME_LABEL_RE.findall(text)),
        "time_label_sequence": TIME_LABEL_RE.findall(text),
        "recurrence": Counter(
            {concept: count for concept, count in concept_occurrences.items() if count > 1}
        ),
        "trends": trends or inferred_trends,
        "facts_by_concept": facts_by_concept,
        "units_by_concept": units_by_concept,
        "format_violations": format_violations,
    }


def _counter_retention(source: Counter[str], summary: Counter[str]) -> tuple[int, int]:
    denominator = sum(source.values())
    retained = sum(min(count, summary.get(key, 0)) for key, count in source.items())
    return retained, denominator


def balanced_fidelity_sample(
    records: Sequence[Mapping[str, Any]],
    *,
    histories_per_task_condition: int,
) -> list[dict[str, Any]]:
    """Select the frozen validation-only 4-task by 2-condition fidelity sample."""
    eligible = [dict(row) for row in records if row.get("split") == "validation"]
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        groups[(str(row["task"]), str(row["condition"]))].append(row)
    if len(groups) != 8:
        raise ContractError("M2 fidelity requires four tasks crossed with correct and shuffled time.")
    sample: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda row: stable_hash((key, row["instance_key"])))
        if len(ranked) < histories_per_task_condition:
            raise ContractError("M2 fidelity group is too small for its prespecified balanced sample.")
        sample.extend(ranked[:histories_per_task_condition])
    return sample


def build_manual_fidelity_packet(
    records: Sequence[Mapping[str, Any]],
    *,
    histories_per_task_condition: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a private review packet and a fillable decision template without patient identity."""
    sample = balanced_fidelity_sample(
        records,
        histories_per_task_condition=histories_per_task_condition,
    )
    entries = []
    decisions = []
    for row in sample:
        review_id = stable_hash(
            {
                "schema": "m2_manual_fidelity_v1",
                "task": row["task"],
                "condition": row["condition"],
                "instance_key": row["instance_key"],
                "source_hash": stable_hash(str(row["source_text"])),
                "summary_hash": stable_hash(str(row["summary"])),
            }
        )
        entries.append(
            {
                "review_id": review_id,
                "task": row["task"],
                "condition": row["condition"],
                "source_text": row["source_text"],
                "summary": row["summary"],
            }
        )
        decisions.append(
            {
                "review_id": review_id,
                "task": row["task"],
                "condition": row["condition"],
                **{criterion: None for criterion in MANUAL_REVIEW_CRITERIA},
            }
        )
    packet = {
        "schema_version": 1,
        "privacy": "GOVERNED_PRIVATE_DO_NOT_COPY_OFF_DGX",
        "instructions": (
            "Compare each source_text with its summary. Set every decision field to true or false; "
            "do not add patient facts or free-text notes."
        ),
        "entries": entries,
    }
    template = {
        "schema_version": 1,
        "privacy": "GOVERNED_PRIVATE_DO_NOT_COPY_OFF_DGX",
        "review_complete": False,
        "decisions": decisions,
    }
    return packet, template


def validate_manual_fidelity_decisions(
    packet: Mapping[str, Any],
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a privacy-safe aggregate; any incomplete or failed item makes M2 ineligible."""
    entries = packet.get("entries")
    reviewed = decisions.get("decisions")
    if not isinstance(entries, list) or not isinstance(reviewed, list) or len(entries) != 40:
        raise ContractError("Manual M2 fidelity review must cover the frozen 40-history packet.")
    if decisions.get("review_complete") is not True or len(reviewed) != len(entries):
        raise ContractError("Manual M2 fidelity decisions are incomplete.")
    expected = {str(row["review_id"]): row for row in entries}
    observed: dict[str, Mapping[str, Any]] = {}
    for row in reviewed:
        if not isinstance(row, dict) or str(row.get("review_id", "")) in observed:
            raise ContractError("Manual M2 fidelity decisions contain a duplicate or invalid review ID.")
        review_id = str(row["review_id"])
        if review_id not in expected:
            raise ContractError("Manual M2 fidelity decisions do not match the frozen packet.")
        if row.get("task") != expected[review_id]["task"] or row.get("condition") != expected[review_id]["condition"]:
            raise ContractError("Manual M2 fidelity task/condition metadata drifted.")
        if any(type(row.get(criterion)) is not bool for criterion in MANUAL_REVIEW_CRITERIA):
            raise ContractError("Every manual M2 fidelity criterion must be explicitly true or false.")
        observed[review_id] = row
    if set(observed) != set(expected):
        raise ContractError("Manual M2 fidelity decisions omit a frozen review item.")
    failures = {
        criterion: sum(not bool(row[criterion]) for row in observed.values())
        for criterion in MANUAL_REVIEW_CRITERIA
    }
    groups = {(str(row["task"]), str(row["condition"])) for row in entries}
    passed = all(value == 0 for value in failures.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "manual_review_completed": True,
        "validation_only": True,
        "sample_histories": len(entries),
        "task_condition_groups": len(groups),
        "criteria_failure_counts": failures,
        "held_out_prompt_refinement": False,
        "patient_level_content_in_aggregate": False,
    }


def audit_summary_fidelity(
    records: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
    histories_per_task_condition: int,
) -> dict[str, Any]:
    """Audit a deterministic balanced validation sample without exposing its rows."""
    sample = balanced_fidelity_sample(
        records,
        histories_per_task_condition=histories_per_task_condition,
    )
    totals = Counter()
    retained = Counter()
    unsupported = 0
    contradictions = 0
    timing_repairs = 0
    format_violations = 0
    failures = 0
    ratios: list[float] = []
    for row in sample:
        source = str(row["source_text"])
        summary = str(row.get("summary", ""))
        if not summary:
            failures += 1
            continue
        ratios.append(len(summary) / max(1, len(source)))
        source_features = _feature_sets(source)
        summary_features = _feature_sets(summary)
        for name in ("concepts", "numbers", "units", "recurrence", "time_labels"):
            kept, total = _counter_retention(source_features[name], summary_features[name])
            retained[name] += kept
            totals[name] += total
        expected_trends = source_features["trends"]
        observed_trends = {word for word in TREND_WORDS if word in summary.casefold()}
        retained["trends"] += len(expected_trends & observed_trends)
        totals["trends"] += len(expected_trends)
        unsupported += sum(
            max(0, count - source_features["concepts"].get(key, 0))
            for key, count in summary_features["concepts"].items()
        )
        unsupported += sum(
            max(0, count - source_features["numbers"].get(key, 0))
            for key, count in summary_features["numbers"].items()
        )
        contradictions += int("contradiction" in summary.casefold())
        for concept, summary_facts in summary_features["facts_by_concept"].items():
            source_facts = source_features["facts_by_concept"].get(concept, set())
            if source_facts:
                contradictions += len(summary_facts - source_facts)
        for concept, count in summary_features["recurrence"].items():
            source_count = source_features["recurrence"].get(concept)
            if source_count is not None and count != source_count:
                contradictions += 1
        format_violations += int(summary_features["format_violations"])
        if "C3_shuffled" in str(row["condition"]):
            source_labels = source_features["time_label_sequence"]
            summary_labels = summary_features["time_label_sequence"]
            timing_repairs += int(summary_labels != source_labels)
    rates = {
        name: retained[name] / totals[name] if totals[name] else 1.0
        for name in ("concepts", "numbers", "units", "recurrence", "trends", "time_labels")
    }
    unsupported_denominator = max(1, totals["concepts"] + totals["numbers"])
    unsupported_rate = unsupported / unsupported_denominator
    failure_rate = failures / len(sample)
    passed = all(
        (
            rates["concepts"] >= float(thresholds["minimum_concept_retention"]),
            rates["numbers"] >= float(thresholds["minimum_numeric_retention"]),
            rates["units"] >= float(thresholds["minimum_unit_retention"]),
            rates["recurrence"] >= float(thresholds["minimum_recurrence_retention"]),
            rates["trends"] >= float(thresholds["minimum_trend_retention"]),
            rates["time_labels"] >= float(thresholds["minimum_time_label_retention"]),
            unsupported_rate <= float(thresholds["maximum_unsupported_fact_rate"]),
            contradictions <= int(thresholds["maximum_contradictions"]),
            timing_repairs <= int(thresholds["maximum_timing_repairs"]),
            format_violations <= int(thresholds["maximum_format_violations"]),
            failure_rate <= float(thresholds["maximum_generation_failure_rate"]),
        )
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "validation_only": True,
        "sample_histories": len(sample),
        "task_condition_groups": 8,
        "histories_per_task_condition": histories_per_task_condition,
        "retention": rates,
        "unsupported_fact_rate": unsupported_rate,
        "contradictions": contradictions,
        "falsified_timing_repairs": timing_repairs,
        "structured_format_violations": format_violations,
        "compression_ratio": {
            "mean": float(np.mean(ratios)) if ratios else None,
            "minimum": float(np.min(ratios)) if ratios else None,
            "maximum": float(np.max(ratios)) if ratios else None,
        },
        "generation_failure_rate": failure_rate,
        "held_out_prompt_refinement": False,
    }
