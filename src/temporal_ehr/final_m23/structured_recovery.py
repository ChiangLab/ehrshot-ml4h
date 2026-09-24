"""DGX-only engineering release gates for ordered structured M2; no clinical fit."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import load_final_m1_authorization
from temporal_ehr.final_m1.embedding import condition_label
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.final_m23.m2 import LocalSummaryTokenValidator, NonRetryableSummaryError
from temporal_ehr.final_m23.protocol import load_final_m23_authorization, require_prompt4_hard_gates
from temporal_ehr.final_m23.recurrence_recovery import CONDITIONS, _sentinel, preserved_evidence
from temporal_ehr.final_m23.structured_summary import (
    VERSION,
    SummaryContractError,
    compact,
    generation_identity,
    ordered_source,
    render,
    request_payload,
    strict_json,
    synthetic_source,
    validate,
)
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

CONFIG = Path("configs/experiments/m2_structured_v1.yaml")
BASELINE = Path("configs/experiments/final_m2_m3.yaml")
PRIVATE = Path("outputs/private/recovery/m2_structured_v1")
PUBLIC = Path("reports/generated/engineering/m2_structured_v1")
OLD_COMMIT = "0d6ad24cbc86bc865542c96d65c665a0ab8a673e"
OLD_MANIFEST_HASH = "b42ed652abbaf911356664589eb52a795a1b2908d12ee78b535a604569eb92c9"
BASELINE_HASH = "758c67c103b71900432deeabcf26744eecaa60f8d83fe200832e6ea7834dadf8"
TASKS = ("lab_anemia", "guo_readmission", "new_hypertension", "chexpert")


def write_once(path: Path, payload: dict) -> None:
    """Atomic no-clobber publication, including across competing writers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read(path: Path) -> dict:
    result = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ContractError("Expected a JSON object in engineering evidence.")
    return result


def load_config(root: Path) -> dict:
    config = yaml.safe_load((root / CONFIG).read_text())
    expected = {
        "version": VERSION, "authorized_on": "2026-09-22", "mode": "VALIDATION_ONLY_ENGINEERING",
        "baseline_config_sha256": BASELINE_HASH, "instruction_model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "api_base": "http://10.45.177.17:8000/v1", "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "embedding_max_tokens": 8192, "source_history_policy": "m1_content_matched_8192",
        "decoding": {"temperature": 0.0, "top_p": 1.0, "seed": 20260815, "max_tokens": 16384,
                     "chat_template_kwargs": {"enable_thinking": False}},
        "execution": {"concurrency": 4, "attempts": 2, "timeout_seconds": 600,
                      "deadline_seconds": 12600, "consecutive_failure_limit": 4},
        "sampling": {"development_per_task_condition": 4, "confirmation_per_task_condition": 4,
                     "seed": 20260922},
        "full_run_authorized": False, "test_data_authorized": False,
    }
    if config != expected or file_sha256(root / BASELINE) != BASELINE_HASH:
        raise ContractError("Structured M2 configuration or immutable baseline drifted.")
    return config


def verify_manifest(directory: Path, expected_hash: str) -> dict:
    path = directory / "manifest.json"
    if file_sha256(path) != expected_hash:
        raise ContractError("Private evidence manifest changed.")
    manifest = read(path)
    actual = {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.json"))
              if p != path}
    if actual != manifest:
        raise ContractError("Private evidence contents changed.")
    return manifest


def old_evidence(root: Path) -> dict:
    """Verify, never promote or rewrite, the failed job-980 evidence."""
    status = _sentinel(root, "980", "FAILED", OLD_COMMIT)
    report_path = root / "reports/generated/engineering/m2_after596/canary-980.json"
    report = read(report_path)
    if (report.get("status") != "FAIL" or report.get("planned_inputs") != 32
            or report.get("recovery_git_commit") != OLD_COMMIT
            or report.get("private_manifest_sha256") != OLD_MANIFEST_HASH
            or report.get("config_sha256") != BASELINE_HASH):
        raise ContractError("Wrong source canary evidence.")
    directory = root / "outputs/private/recovery/m2_after596/980"
    verify_manifest(directory, OLD_MANIFEST_HASH)
    original = preserved_evidence(root)
    if stable_hash(original) != report["original_evidence_hash"]:
        raise ContractError("Original 596 or M3 evidence changed.")
    return {"old_failed_status_hash": file_sha256(status), "old_report_hash": file_sha256(report_path),
            "old_private_manifest_hash": OLD_MANIFEST_HASH, "older_evidence_hash": stable_hash(original)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("endpoint_redirect_forbidden")


class PrivateClient:
    """No external proxy or redirect; never echo a remote exception body."""

    def __init__(self, config: dict):
        if config["api_base"] != "http://10.45.177.17:8000/v1":
            raise ContractError("Only the authorized local endpoint is permitted.")
        self.config = config
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path: str, payload: dict | None = None) -> dict:
        if path not in {"models", "chat/completions"}:
            raise ContractError("Endpoint route not allowed.")
        request = urllib.request.Request(self.config["api_base"] + "/" + path,
                                         data=compact(payload).encode() if payload is not None else None,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.config["execution"]["timeout_seconds"]) as response:
                body = response.read(8_000_001)
            if len(body) > 8_000_000:
                raise RuntimeError("endpoint_response_oversized")
            result = strict_json(body.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, UnicodeError, SummaryContractError) as exc:
            raise RuntimeError("endpoint_transport_or_json_failure") from exc
        if not isinstance(result, dict):
            raise RuntimeError("endpoint_nonobject_response")
        return result

    def identity(self) -> str:
        models = self.request("models").get("data", [])
        record = next((m for m in models if m.get("id") == self.config["instruction_model"]), None)
        if record is None:
            raise ContractError("Locked instruction model is not served.")
        return stable_hash({k: record.get(k) for k in ("id", "root", "parent", "owned_by", "max_model_len")})


def response_document(response: dict) -> dict:
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") != "stop":
            raise SummaryContractError("completion_not_stop")
        if not isinstance(content, str) or not content.strip():
            raise SummaryContractError("empty_or_nonstring_content")
        return strict_json(content)
    except (KeyError, IndexError, TypeError) as exc:
        raise SummaryContractError("missing_completion_metadata") from exc


def assess(item: dict, config: dict, thresholds: dict, counter: Any, directory: Path,
           identity: str, client: Any = None) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    write_once(directory / "source.json", item)
    client = client or PrivateClient(config)
    errors, response_hashes = [], set()
    started = time.monotonic()
    for attempt in range(config["execution"]["attempts"]):
        try:
            response = client.request("chat/completions", request_payload(item["source"], config))
            write_once(directory / f"response-{attempt + 1}.json", response)
            document = response_document(response)
            digest = stable_hash(document)
            if digest in response_hashes:
                raise SummaryContractError("identical_rejected_response")
            response_hashes.add(digest)
            validation = validate(document, item["source"], thresholds=thresholds, token_counter=counter)
            result = {"status": "PASS", "document": document, "summary": render(document),
                      "validation": validation, "attempts": attempt + 1, "errors": errors,
                      "generation_key": generation_identity(item["source"], item["condition"], config, identity),
                      "seconds": time.monotonic() - started}
            write_once(directory / "assessment.json", result)
            return result
        except SummaryContractError as exc:
            errors.append(str(exc))
        except NonRetryableSummaryError:
            errors.append("embedding_token_budget")
        except RuntimeError:
            errors.append("endpoint_failure")
        # A fixed bounded retry, never an adaptive prompt or recursive summary.
        if attempt + 1 < config["execution"]["attempts"]:
            time.sleep(1)
    result = {"status": "FAIL", "errors": errors, "seconds": time.monotonic() - started}
    write_once(directory / "assessment.json", result)
    return result


def select_samples(candidates: list[dict], old_subjects: set[str], tasks: tuple, seed: int = 20260922) -> dict:
    """Freeze both samples before generation, disjoint across people, not just texts."""
    if any(row.get("split") != "train" for row in candidates):
        raise ContractError("Engineering samples must be training-only.")
    selected: dict[str, list] = {"development": [], "confirmation": []}
    development_subjects = set(old_subjects)
    for phase in selected:
        for task in tasks:
            for condition in CONDITIONS:
                rows = [r for r in candidates if r["task"] == task and r["condition"] == condition
                        and (phase == "development" or r["person_hash"] not in development_subjects)]
                # One official instance per patient/task; remove accidental duplicates.
                unique = {r["person_hash"]: r for r in rows}
                rows = sorted(unique.values(), key=lambda r: (
                    len(compact(r["source"])), stable_hash({"seed": seed, "source": r["source"],
                                                          "person": r["person_hash"]})))
                if len(rows) < 4:
                    raise ContractError("Insufficient disjoint training examples for engineering gate.")
                indices = (0, (len(rows) - 1) // 3, 2 * (len(rows) - 1) // 3, len(rows) - 1)
                chosen = [rows[i] for i in indices]
                selected[phase].extend(chosen)
                if phase == "development":
                    development_subjects.update(r["person_hash"] for r in chosen)
    return selected


def verify_samples(plan: dict) -> None:
    """Check training-only balance and person-disjoint confirmation, not just row counts."""
    expected = Counter({(task, condition): 4 for task in TASKS for condition in CONDITIONS})
    subjects = {}
    for phase in ("development", "confirmation"):
        rows = plan["samples"][phase]
        if (any(r.get("split") != "train" for r in rows)
                or Counter((r["task"], r["condition"]) for r in rows) != expected
                or len({(r["task"], r["condition"], r["person_hash"]) for r in rows}) != 32):
            raise ContractError("Engineering sample is unbalanced, duplicated, or not training-only.")
        subjects[phase] = {r["person_hash"] for r in rows}
    if subjects["confirmation"] & (subjects["development"] | set(plan["prior_canary_people"])):
        raise ContractError("Confirmation people overlap prior engineering inputs.")


def legacy_text_for_exclusion(source: dict) -> str:
    """Reconstruct old grouping ONLY to exclude job-980 people from confirmation.

    This deliberately lossy representation must never be passed to new generation.
    """
    if not source["windows"]:
        return "[NO_PRE_T_EVENTS]"
    lines = []
    for window in source["windows"]:
        lines.append(window["label"])
        for fact in window["facts"]:
            parts = [f'- {fact["code"]}']
            for field, name in (("value", "values"), ("text", "text_values")):
                values = list(dict.fromkeys(r[field] for r in fact["sequence"] if r[field] is not None))
                if values:
                    parts.append(f"{name}=" + (",".join(values) if field == "value" else compact(values)))
            for field in ("unit", "source"):
                if fact[field] is not None:
                    parts.append(f"{field}={fact[field]}")
            parts.append("occurrences=" + str(sum(r["count"] for r in fact["sequence"])))
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def prepare(root: Path, directory: Path, config: dict, commit: str, identity: str, tokenizer_hash: str) -> dict:
    require_prompt4_hard_gates(root, root / BASELINE)
    evidence = old_evidence(root)
    spec = load_final_m23_authorization(root / BASELINE, repository_root=root)
    m1 = load_final_m1_authorization(spec.m1_config_path, repository_root=root)
    old_items = read(root / "outputs/private/recovery/m2_after596/980/inputs.json")["items"]
    old_hashes = {stable_hash(item["source_text"]) for item in old_items}
    candidates, old_subjects, upstream_hashes, matched_old = [], set(), {}, set()
    for task in spec.tasks:
        rows = read_final_task_texts(m1, task)
        upstream_hashes[task] = file_sha256(m1.private_root / "prepared" / task / "texts.parquet")
        for row in rows:
            if row["split"] not in {"train", "validation"}:
                raise ContractError("Prepared source contains an unauthorized split.")
            if row["split"] != "train" or condition_label(row) not in CONDITIONS:
                continue
            text = str(row["text"])
            person = stable_hash(str(row["subject_id"]))
            source = ordered_source(text, m1.instructions[task])
            legacy_hash = stable_hash(legacy_text_for_exclusion(source))
            if legacy_hash in old_hashes:
                old_subjects.add(person)
                matched_old.add(legacy_hash)
            candidates.append({"task": task, "condition": condition_label(row), "split": "train",
                               "person_hash": person, "prepared_text_hash": stable_hash(text),
                               "source": source})
    if matched_old != old_hashes:
        raise ContractError("Could not resolve every prior canary source for independent-sample exclusion.")
    samples = select_samples(candidates, old_subjects, spec.tasks, config["sampling"]["seed"])
    plan = {"version": VERSION, "config_hash": stable_hash(config), "source_commit": commit,
            "served_identity": identity, "tokenizer_hash": tokenizer_hash, "evidence": evidence,
            "upstream_hashes": upstream_hashes, "prior_canary_people": sorted(old_subjects), "samples": samples}
    verify_samples(plan)
    write_once(directory / "plan.json", plan)
    return plan


def tokenizer_fingerprint(model_root: Path) -> str:
    directory = model_root / "Qwen3-Embedding-0.6B"
    return stable_hash({name: file_sha256(directory / name) for name in (
        "tokenizer.json", "tokenizer_config.json", "config.json", "PINNED_REVISION.txt",
        "MODEL_ARTIFACT_MANIFEST.json")})


def verify_upstream(root: Path, plan: dict) -> None:
    for task, expected in plan["upstream_hashes"].items():
        if task not in {"lab_anemia", "guo_readmission", "new_hypertension", "chexpert"}:
            raise ContractError("Unexpected upstream task.")
        path = root / "outputs/private/final/locked_four_task_m1_v1/prepared" / task / "texts.parquet"
        if file_sha256(path) != expected:
            raise ContractError("Immutable M1 prepared source changed.")


def verify_phase(root: Path, job: str, phase: str, commit: str, counter: Any) -> dict:
    config = load_config(root)
    _sentinel(root, job, "COMPLETE", commit)
    report = read(root / PUBLIC / f"{phase}-{job}.json")
    directory = root / PRIVATE / job
    if (report.get("status") != "PASS" or report.get("source_commit") != commit
            or report.get("phase") != phase or report.get("job_id") != job
            or report.get("config_hash") != stable_hash(config) or report.get("checked") != 32
            or report.get("full_run_authorized") is not False or report.get("test_data_used") is not False):
        raise ContractError("Structured engineering report is stale or incomplete.")
    verify_manifest(directory, report["private_manifest_hash"])
    plan = read(directory / "plan.json")
    verify_samples(plan)
    verify_upstream(root, plan)
    if (plan["evidence"] != old_evidence(root) or plan["config_hash"] != stable_hash(config)
            or plan["source_commit"] != commit or plan["served_identity"] != report["served_identity"]
            or plan["tokenizer_hash"] != report["tokenizer_hash"]):
        raise ContractError("Frozen engineering inputs or historical evidence changed.")
    if phase == "confirmation":
        parent = read(directory / "parent.json")
        if not isinstance(parent.get("job"), str) or not re.fullmatch(r"[0-9]+", parent["job"]):
            raise ContractError("Invalid development parent identity.")
        parent_report = verify_phase(root, parent["job"], "development", commit, counter)
        if (stable_hash(parent_report) != parent["report_hash"]
                or read(root / PRIVATE / parent["job"] / "plan.json") != plan):
            raise ContractError("Confirmation does not match the completed frozen development plan.")
    thresholds = yaml.safe_load((root / BASELINE).read_text())["m2"]["fidelity"]
    if len(plan["samples"][phase]) != 32:
        raise ContractError("Wrong engineering sample size.")
    probe = read(directory / "synthetic/assessment.json")
    if probe.get("status") != "PASS" or validate(
        probe["document"], synthetic_source(), thresholds=thresholds, token_counter=counter
    ) != probe["validation"]:
        raise ContractError("Synthetic schema/fidelity probe did not pass.")
    for index, item in enumerate(plan["samples"][phase]):
        item_directory = directory / f"item-{index:02d}"
        result = read(item_directory / "assessment.json")
        if item["split"] != "train" or result.get("status") != "PASS":
            raise ContractError("Incomplete or nontraining engineering item.")
        attempts = result.get("attempts")
        if (type(attempts) is not int or not 1 <= attempts <= config["execution"]["attempts"]
                or read(item_directory / "source.json") != item
                or response_document(read(item_directory / f"response-{attempts}.json")) != result["document"]):
            raise ContractError("Assessment is not bound to the retained source and actual model response.")
        actual = validate(result["document"], item["source"], thresholds=thresholds, token_counter=counter)
        if (actual != result["validation"] or render(result["document"]) != result["summary"]
                or result["generation_key"] != generation_identity(
                    item["source"], item["condition"], config, plan["served_identity"])):
            raise ContractError("Structured assessment failed semantic revalidation.")
    return report


def run(root: Path, model_root: Path, job: str, phase: str, commit: str, parent: str | None = None) -> dict:
    config = load_config(root)
    directory = root / PRIVATE / job
    directory.mkdir(parents=True, exist_ok=False)
    counter = LocalSummaryTokenValidator(model_root / "Qwen3-Embedding-0.6B", maximum_tokens=8192)
    thresholds = yaml.safe_load((root / BASELINE).read_text())["m2"]["fidelity"]
    client = PrivateClient(config)
    identity = client.identity()
    tokenizer_hash = tokenizer_fingerprint(model_root)
    if phase == "development":
        plan = prepare(root, directory, config, commit, identity, tokenizer_hash)
    else:
        if parent is None:
            raise ContractError("Independent confirmation requires a development job.")
        verified = verify_phase(root, parent, "development", commit, counter)
        plan = read(root / PRIVATE / parent / "plan.json")
        if identity != plan["served_identity"] or tokenizer_hash != plan["tokenizer_hash"]:
            raise ContractError("Serving or tokenizer identity changed between engineering stages.")
        write_once(directory / "plan.json", plan)
        write_once(directory / "parent.json", {"job": parent, "report_hash": stable_hash(verified)})
    started = time.monotonic()
    probe = assess({"source": synthetic_source(), "condition": "synthetic"}, config, thresholds,
                   counter, directory / "synthetic", identity)
    results, stopped = [], "none"
    if probe["status"] == "PASS":
        items = plan["samples"][phase]
        failures_in_a_row = 0
        # Bounded waves: no unbounded pending-future queue after a circuit breaker trips.
        with ThreadPoolExecutor(max_workers=config["execution"]["concurrency"]) as pool:
            for start in range(0, len(items), config["execution"]["concurrency"]):
                if time.monotonic() - started > config["execution"]["deadline_seconds"] - 1300:
                    stopped = "deadline_reserve"
                    break
                wave = list(enumerate(items))[start:start + config["execution"]["concurrency"]]
                futures = [pool.submit(assess, item, config, thresholds, counter,
                                       directory / f"item-{i:02d}", identity) for i, item in wave]
                for future in futures:
                    result = future.result()
                    results.append(result)
                    failures_in_a_row = failures_in_a_row + 1 if result["status"] == "FAIL" else 0
                    print(json.dumps({"event": "structured_m2_progress", "phase": phase,
                                      "checked": len(results), "planned": 32}), flush=True)
                if failures_in_a_row >= config["execution"]["consecutive_failure_limit"]:
                    stopped = "failure_circuit_breaker"
                    break
    else:
        stopped = "synthetic_schema_or_fidelity_failure"
    verify_upstream(root, plan)
    if (old_evidence(root) != plan["evidence"] or client.identity() != identity
            or tokenizer_fingerprint(model_root) != tokenizer_hash):
        raise ContractError("Source evidence or served identity changed during generation.")
    passed = len(results) == 32 and all(r["status"] == "PASS" for r in results)
    errors = Counter(error for result in [probe, *results] for error in result["errors"])
    manifest = {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.json"))}
    write_once(directory / "manifest.json", manifest)
    report = {"version": VERSION, "phase": phase, "job_id": job, "source_commit": commit,
              "status": "PASS" if passed else "FAIL", "config_hash": stable_hash(config),
              "served_identity": identity, "tokenizer_hash": tokenizer_hash, "checked": len(results),
              "failure_categories": {k: n if n >= 20 else "<20" for k, n in sorted(errors.items())},
              "private_manifest_hash": file_sha256(directory / "manifest.json"), "stop_reason": stopped,
              "runtime_seconds": time.monotonic() - started, "full_run_authorized": False,
              "test_data_used": False, "embedding_model_run": False, "scientific_M2_complete": False,
              "manual_fidelity_review_complete": False,
              "interpretation": "Training-only engineering evidence; inspect before production scheduling."}
    valid = [r["validation"] for r in results if r["status"] == "PASS"]
    # Aggregate-only feasibility statistics; suppress small successful subsets.
    if len(valid) >= 20:
        report["feasibility"] = {
            "maximum_embedding_tokens": max(r["embedding_tokens"] for r in valid),
            "maximum_character_ratio_to_grouped_source": max(r["character_compression_ratio"] for r in valid),
            "minimum_retention": {key: min(r["retention"][key] for r in valid) for key in valid[0]["retention"]},
        }
    assert_privacy_safe_report(json.dumps(report))
    write_once(root / PUBLIC / f"{phase}-{job}.json", report)
    return report


def check_runtime(root: Path, job: str, running: bool) -> str:
    if root != Path("/data/shared73/projects/wdelery/temporal-ehr-ml4h") or not os.environ.get("SLURM_JOB_ID"):
        raise ContractError("Run only inside the declared DGX Slurm container.")
    if not re.fullmatch(r"[0-9]+", job) or (running and job != os.environ["SLURM_JOB_ID"]):
        raise ContractError("Wrong job identity.")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ContractError("Engineering source checkout must be clean.")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
