"""Replay exact job-982 responses and confirm compact rendering on the frozen sample."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_render_v2 as compact_render
from temporal_ehr.final_m23.m2 import NonRetryableSummaryError
from temporal_ehr.final_m23.structured_summary import (
    SummaryContractError,
    generation_identity,
    synthetic_source,
)
from temporal_ehr.final_m23.structured_summary import (
    render as render_v1,
)
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

CONFIG = Path("configs/recovery/m2_render_v2_after_982.yaml")
PRIVATE = Path("outputs/private/recovery/m2_render_v2")
PUBLIC = Path("reports/generated/engineering/m2_render_v2")
EXPECTED = {
    "version": compact_render.VERSION, "source_job_id": "982", "unused_confirmation_job_id": "983",
    "source_commit": "f9cce3d09196177ab123e5467854f0435773c051",
    "source_private_manifest_sha256": "480effa80c708f4e04e29fdb863cf1f80cc3f49048da232c820f90c27ce120d1",
    "generation_module_sha256": "a5754c13cf081240bffedcb89af4ea561211776e9a2202281ccf268635216269",
    "generation_config_sha256": "5e09a39b802f865b9aa5ed160bb4b3032301ed421bafa07b8dac3d9b8dd4d1d5",
    "selection": "last_saved_response_per_development_input",
    "full_run_authorized": False, "test_data_authorized": False,
}


def source_evidence(root: Path) -> tuple[dict, dict]:
    repair = yaml.safe_load((root / CONFIG).read_text())
    if repair != EXPECTED:
        raise ContractError("Rendering repair authorization changed.")
    for relative, key in ((old.CONFIG, "generation_config_sha256"),
                          (Path("src/temporal_ehr/final_m23/structured_summary.py"), "generation_module_sha256")):
        if file_sha256(root / relative) != repair[key]:
            raise ContractError("Original generation contract changed.")
    config = old.load_config(root)
    status = old._sentinel(root, "982", "FAILED", repair["source_commit"])
    report_path = root / old.PUBLIC / "development-982.json"
    report = old.read(report_path)
    directory = root / old.PRIVATE / "982"
    if (report.get("status") != "FAIL" or report.get("checked") != 32
            or report.get("phase") != "development" or report.get("job_id") != "982"
            or report.get("source_commit") != repair["source_commit"]
            or report.get("config_hash") != stable_hash(config)
            or report.get("private_manifest_hash") != repair["source_private_manifest_sha256"]
            or report.get("embedding_model_run") is not False or report.get("test_data_used") is not False):
        raise ContractError("Job 982 is not the declared failed development evidence.")
    old.verify_manifest(directory, repair["source_private_manifest_sha256"])
    plan = old.read(directory / "plan.json")
    old.verify_samples(plan)
    old.verify_upstream(root, plan)
    if (plan["config_hash"] != stable_hash(config) or plan["source_commit"] != repair["source_commit"]
            or plan["evidence"] != old.old_evidence(root)
            or plan["served_identity"] != report["served_identity"]
            or plan["tokenizer_hash"] != report["tokenizer_hash"]):
        raise ContractError("Original plan identity or evidence mismatch.")
    # Confirmation independence is not assumed if the original job ever started.
    if (root / old.PRIVATE / "983").exists() or any(
        (root / "outputs/private/run_status/983" / f"{state}.json").exists()
        for state in ("RUNNING", "COMPLETE")
    ):
        raise ContractError("Original confirmation may have started; independent sample review required.")
    return plan, {"failed_sentinel_sha256": file_sha256(status), "report_sha256": file_sha256(report_path),
                  "private_manifest_sha256": repair["source_private_manifest_sha256"],
                  "plan_sha256": file_sha256(directory / "plan.json")}


def seal_result(result: dict, item: dict, config: dict, identity: str, tokenizer_hash: str) -> dict:
    if result["status"] == "PASS":
        result["generation_key"] = generation_identity(item["source"], item["condition"], config, identity)
        result["embedding_input_key"] = compact_render.embedding_identity(
            result["generation_key"], result["summary"], tokenizer_hash)
    return result


def replay_item(root: Path, index: int, item: dict, destination: Path, thresholds: dict,
                counter: Any, config: dict, plan: dict) -> dict:
    original = root / old.PRIVATE / "982" / f"item-{index:02d}"
    if old.read(original / "source.json") != item:
        raise ContractError("Saved response source does not match frozen sample.")
    responses = sorted(original.glob("response-*.json"))
    if not responses or any(p.name not in {"response-1.json", "response-2.json"} for p in responses):
        raise ContractError("Unexpected saved response inventory.")
    response = old.read(responses[-1])
    destination.mkdir(parents=True, exist_ok=False)
    old.write_once(destination / "source.json", item)
    old.write_once(destination / "response-1.json", response)
    old.write_once(destination / "origin.json", {"job": "982", "response_file": responses[-1].name,
                                                "response_sha256": file_sha256(responses[-1])})
    try:
        document = old.response_document(response)
        validation = compact_render.validate(document, item["source"], thresholds=thresholds, token_counter=counter)
        result = {"status": "PASS", "document": document, "summary": compact_render.render(document),
                  "validation": validation, "attempts": 1, "errors": [], "new_generation": False,
                  "previous_render_tokens": counter.count(render_v1(document))}
    except (SummaryContractError, NonRetryableSummaryError) as exc:
        error = "embedding_token_budget" if isinstance(exc, NonRetryableSummaryError) else str(exc)
        result = {"status": "FAIL", "errors": [error], "new_generation": False}
    result = seal_result(result, item, config, plan["served_identity"], plan["tokenizer_hash"])
    old.write_once(destination / "assessment.json", result)
    return result


def run(root: Path, model_root: Path, job: str, phase: str, commit: str, parent: str | None = None) -> dict:
    plan, evidence = source_evidence(root)
    config = old.load_config(root)
    counter = old.LocalSummaryTokenValidator(model_root / "Qwen3-Embedding-0.6B", maximum_tokens=8192)
    if old.tokenizer_fingerprint(model_root) != plan["tokenizer_hash"]:
        raise ContractError("Tokenizer differs from the original generation run.")
    thresholds = yaml.safe_load((root / old.BASELINE).read_text())["m2"]["fidelity"]
    directory = root / PRIVATE / job
    directory.mkdir(parents=True, exist_ok=False)
    old.write_once(directory / "plan.json", plan)
    old.write_once(directory / "evidence.json", evidence)
    started = time.monotonic()
    results, stop_reason = [], "none"
    if phase == "replay":
        for index, item in enumerate(plan["samples"]["development"]):
            results.append(replay_item(root, index, item, directory / f"item-{index:02d}",
                                       thresholds, counter, config, plan))
            print(json.dumps({"event": "m2_render_replay", "checked": len(results), "planned": 32}), flush=True)
    elif phase == "confirmation":
        if parent is None:
            raise ContractError("Confirmation requires a completed replay parent.")
        parent_report = verify(root, model_root, parent, "replay", commit)
        old.write_once(directory / "parent.json", {"job": parent, "report_hash": stable_hash(parent_report)})
        client = old.PrivateClient(config)
        if client.identity() != plan["served_identity"]:
            raise ContractError("Served model metadata changed since job 982.")
        probe = old.assess({"source": synthetic_source(), "condition": "synthetic"}, config, thresholds,
                           counter, directory / "synthetic", plan["served_identity"], client,
                           validator=compact_render.validate, renderer=compact_render.render)
        if probe["status"] == "PASS":
            items = plan["samples"]["confirmation"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for start in range(0, 32, 4):
                    if time.monotonic() - started > config["execution"]["deadline_seconds"] - 1300:
                        stop_reason = "deadline_reserve"
                        break
                    futures = [pool.submit(old.assess, items[i], config, thresholds, counter,
                                           directory / f"item-{i:02d}", plan["served_identity"],
                                           validator=compact_render.validate, renderer=compact_render.render)
                               for i in range(start, start + 4)]
                    wave = [future.result() for future in futures]
                    for i, result in enumerate(wave, start):
                        sealed = seal_result(result, items[i], config, plan["served_identity"], plan["tokenizer_hash"])
                        # Original per-attempt assessment stays immutable; derived admission is separate.
                        old.write_once(directory / f"item-{i:02d}" / "admission.json", sealed)
                        results.append(sealed)
                    print(json.dumps({"event": "m2_render_confirmation", "checked": len(results),
                                      "planned": 32}), flush=True)
                    if all(r["status"] == "FAIL" for r in wave):
                        stop_reason = "failure_circuit_breaker"
                        break
        else:
            stop_reason = "synthetic_probe_failed"
        if client.identity() != plan["served_identity"]:
            raise ContractError("Served model metadata changed during confirmation.")
    else:
        raise ContractError("Unknown rendering recovery phase.")
    _, evidence_after = source_evidence(root)
    if evidence_after != evidence or old.tokenizer_fingerprint(model_root) != plan["tokenizer_hash"]:
        raise ContractError("Immutable evidence or tokenizer changed during recovery.")
    manifest = {str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.json"))}
    old.write_once(directory / "manifest.json", manifest)
    valid = [r for r in results if r["status"] == "PASS"]
    failures = Counter(error for r in results for error in r["errors"])
    report = {"version": compact_render.VERSION, "phase": phase, "job_id": job,
              "source_generation_job": "982", "source_generation_commit": EXPECTED["source_commit"],
              "recovery_commit": commit, "recovery_config_hash": file_sha256(root / CONFIG),
              "status": "PASS" if len(valid) == 32 else "FAIL", "checked": len(results),
              "generation_config_hash": stable_hash(config), "evidence_hash": stable_hash(evidence),
              "private_manifest_hash": file_sha256(directory / "manifest.json"),
              "tokenizer_hash": plan["tokenizer_hash"], "served_identity": plan["served_identity"],
              "failure_categories": {k: n if n >= 20 else "<20" for k, n in sorted(failures.items())},
              "stop_reason": stop_reason, "runtime_seconds": time.monotonic() - started,
              "new_generations_requested": phase == "confirmation", "test_data_used": False,
              "embedding_model_run": False, "full_run_authorized": False, "scientific_M2_complete": False}
    if len(valid) >= 20:
        tokens = [r["validation"]["embedding_tokens"] for r in valid]
        report["feasibility"] = {"maximum_embedding_tokens": max(tokens), "median_embedding_tokens": median(tokens),
                                 "minimum_retention": {k: min(r["validation"]["retention"][k] for r in valid)
                                                       for k in valid[0]["validation"]["retention"]}}
        if phase == "replay":
            report["feasibility"]["previous_maximum_tokens"] = max(r["previous_render_tokens"] for r in valid)
    assert_privacy_safe_report(json.dumps(report))
    old.write_once(root / PUBLIC / f"{phase}-{job}.json", report)
    return report


def verify(root: Path, model_root: Path, job: str, phase: str, commit: str, *, final: bool = True) -> dict:
    if not re.fullmatch(r"[0-9]+", job) or phase not in {"replay", "confirmation"}:
        raise ContractError("Invalid recovery identity.")
    if final:
        old._sentinel(root, job, "COMPLETE", commit)
    plan, evidence = source_evidence(root)
    config = old.load_config(root)
    directory = root / PRIVATE / job
    report = old.read(root / PUBLIC / f"{phase}-{job}.json")
    if (report.get("status") != "PASS" or report.get("checked") != 32
            or report.get("recovery_commit") != commit or report.get("phase") != phase
            or report.get("job_id") != job or report.get("version") != compact_render.VERSION
            or report.get("recovery_config_hash") != file_sha256(root / CONFIG)
            or report.get("generation_config_hash") != stable_hash(config)
            or report.get("evidence_hash") != stable_hash(evidence)
            or report.get("tokenizer_hash") != old.tokenizer_fingerprint(model_root)
            or report.get("tokenizer_hash") != plan["tokenizer_hash"]
            or report.get("served_identity") != plan["served_identity"]
            or report.get("source_generation_job") != "982"
            or report.get("source_generation_commit") != EXPECTED["source_commit"]
            or report.get("new_generations_requested") is not (phase == "confirmation")
            or report.get("test_data_used") is not False or report.get("embedding_model_run") is not False
            or report.get("full_run_authorized") is not False):
        raise ContractError("Rendering recovery report is incomplete or stale.")
    old.verify_manifest(directory, report["private_manifest_hash"])
    if old.read(directory / "plan.json") != plan or old.read(directory / "evidence.json") != evidence:
        raise ContractError("Original plan/evidence binding changed.")
    counter = old.LocalSummaryTokenValidator(model_root / "Qwen3-Embedding-0.6B", maximum_tokens=8192)
    thresholds = yaml.safe_load((root / old.BASELINE).read_text())["m2"]["fidelity"]
    if phase == "confirmation":
        parent = old.read(directory / "parent.json")
        if stable_hash(verify(root, model_root, parent["job"], "replay", commit)) != parent["report_hash"]:
            raise ContractError("Replay parent report changed.")
        probe = old.read(directory / "synthetic/assessment.json")
        if (probe.get("status") != "PASS" or compact_render.validate(
                probe["document"], synthetic_source(), thresholds=thresholds,
                token_counter=counter) != probe["validation"]):
            raise ContractError("Confirmation synthetic probe invalid.")
    items = plan["samples"]["development" if phase == "replay" else "confirmation"]
    for index, item in enumerate(items):
        item_dir = directory / f"item-{index:02d}"
        result = old.read(item_dir / ("assessment.json" if phase == "replay" else "admission.json"))
        attempt = result.get("attempts")
        if (result.get("status") != "PASS" or type(attempt) is not int or not 1 <= attempt <= 2
                or old.read(item_dir / "source.json") != item):
            raise ContractError("Invalid source/assessment binding.")
        response = old.read(item_dir / f"response-{attempt}.json")
        document = old.response_document(response)
        if phase == "replay":
            original = root / old.PRIVATE / "982" / f"item-{index:02d}"
            chosen = sorted(original.glob("response-*.json"))[-1]
            if old.read(chosen) != response or old.read(item_dir / "origin.json") != {
                "job": "982", "response_file": chosen.name, "response_sha256": file_sha256(chosen)
            }:
                raise ContractError("Replay did not use the declared original response.")
        actual = compact_render.validate(document, item["source"], thresholds=thresholds, token_counter=counter)
        if (document != result["document"] or actual != result["validation"]
                or compact_render.render(document) != result["summary"]
                or result != seal_result(dict(result), item, config, plan["served_identity"], plan["tokenizer_hash"])):
            raise ContractError("Re-rendered response failed semantic/cache-key verification.")
    return report
