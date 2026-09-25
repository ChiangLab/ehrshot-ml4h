"""Evidence-bound 986 repair: regression -> fresh confirmation -> readiness only."""

from __future__ import annotations

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.final_m23 import render_recovery as prior
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_v2 as generation
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

PRIVATE = Path("outputs/private/recovery/m2_structured_v2")
PUBLIC = Path("reports/generated/engineering/m2_structured_v2")
PRIOR_COMMIT = "725162f1a900183952d22b705d335dd5255dbb3a"
PRIOR_MANIFEST = "f0640b4f59ba86362fee5ef1ca44c559ca634632a593b29f592fc205f612b51a"
SAMPLING_SEED = 20260925
PHASES = ("regression", "confirmation", "readiness")


def contract(root: Path) -> dict:
    return {"version": generation.VERSION, "prompt_hash": stable_hash(generation.PROMPT),
            "base_config_hash": stable_hash(old.load_config(root)), "sampling_seed": SAMPLING_SEED,
            "development": "all_64_previously_inspected_982_986_inputs",
            "confirmation": "32_new_training_inputs_excluding_all_prior_engineering_people",
            "execution": {"concurrency": 4, "attempts": 2, "regression_deadline_seconds": 25200,
                          "confirmation_deadline_seconds": 12600, "wave_reserve_seconds": 1800},
            "full_run_authorized": False, "test_data_used": False}


def evidence(root: Path, model_root: Path) -> tuple[dict, dict]:
    """Do not promote 986: verify its immutable failure and the successful 985 replay."""
    replay = prior.verify(root, model_root, "985", "replay", PRIOR_COMMIT)
    failed = old._sentinel(root, "986", "FAILED", PRIOR_COMMIT)
    directory = root / prior.PRIVATE / "986"
    old.verify_manifest(directory, PRIOR_MANIFEST)
    report_path = root / prior.PUBLIC / "confirmation-986.json"
    report = old.read(report_path)
    plan, source_evidence = prior.source_evidence(root)
    if (report.get("status") != "FAIL" or report.get("checked") != 32
            or report.get("job_id") != "986" or report.get("phase") != "confirmation"
            or report.get("private_manifest_hash") != PRIOR_MANIFEST
            or report.get("recovery_commit") != PRIOR_COMMIT
            or report.get("test_data_used") is not False
            or report.get("embedding_model_run") is not False
            or report.get("generation_config_hash") != stable_hash(old.load_config(root))
            or report.get("evidence_hash") != stable_hash(source_evidence)
            or report.get("tokenizer_hash") != plan["tokenizer_hash"]
            or report.get("served_identity") != plan["served_identity"]
            or old.read(directory / "plan.json") != plan
            or old.read(directory / "evidence.json") != source_evidence
            or old.read(directory / "parent.json") != {"job": "985", "report_hash": stable_hash(replay)}):
        raise ContractError("Job 986 does not match the authorized failed evidence.")
    states = Counter()
    for i, item in enumerate(plan["samples"]["confirmation"]):
        path = directory / f"item-{i:02d}"
        if old.read(path / "source.json") != item:
            raise ContractError("Job 986 source binding changed.")
        states[old.read(path / "admission.json")["status"]] += 1
    if states != {"PASS": 29, "FAIL": 3}:
        raise ContractError("Job 986 does not match the observed final input accounting.")
    return plan, {"replay_report_hash": stable_hash(replay), "confirmation_manifest_hash": PRIOR_MANIFEST,
                  "confirmation_report_sha256": file_sha256(report_path),
                  "confirmation_failed_sha256": file_sha256(failed), "prior_evidence": source_evidence}


def candidates(root: Path) -> list[dict]:
    """Use immutable M1-retained texts, never raw histories, outcomes, or test artifacts."""
    old.require_prompt4_hard_gates(root, root / old.BASELINE)
    spec = old.load_final_m23_authorization(root / old.BASELINE, repository_root=root)
    m1 = old.load_final_m1_authorization(spec.m1_config_path, repository_root=root)
    result = []
    for task in spec.tasks:
        for row in old.read_final_task_texts(m1, task):
            if row["split"] not in {"train", "validation"}:
                raise ContractError("Unauthorized split in prepared source.")
            if row["split"] != "train" or old.condition_label(row) not in old.CONDITIONS:
                continue
            text = str(row["text"])
            result.append({"task": task, "condition": old.condition_label(row), "split": "train",
                           "person_hash": stable_hash(str(row["subject_id"])),
                           "prepared_text_hash": stable_hash(text),
                           "source": old.ordered_source(text, m1.instructions[task])})
    return result


def select_confirmation(rows: list[dict], excluded: set[str]) -> list[dict]:
    if any(r.get("split") != "train" for r in rows):
        raise ContractError("Only training inputs may enter confirmation selection.")
    selected = []
    for task in old.TASKS:
        for condition in old.CONDITIONS:
            eligible = [r for r in rows if r["task"] == task and r["condition"] == condition
                        and r["person_hash"] not in excluded]
            if len({r["person_hash"] for r in eligible}) != len(eligible):
                raise ContractError("Ambiguous duplicated engineering candidate.")
            eligible.sort(key=lambda r: (len(old.compact(r["source"])), stable_hash({
                "seed": SAMPLING_SEED, "person": r["person_hash"], "source": r["source"]})))
            if len(eligible) < 4:
                raise ContractError("Insufficient fresh disjoint confirmation candidates.")
            n = len(eligible) - 1
            selected.extend(eligible[i] for i in (0, n // 3, 2 * n // 3, n))
    return selected


def make_plan(root: Path, model_root: Path, commit: str) -> dict:
    original, recorded = evidence(root, model_root)
    regression = original["samples"]["development"] + original["samples"]["confirmation"]
    excluded = set(original["prior_canary_people"]) | {r["person_hash"] for r in regression}
    return {"source_commit": commit, "contract": contract(root), "evidence": recorded,
            "tokenizer_hash": original["tokenizer_hash"], "served_identity": original["served_identity"],
            "upstream_hashes": original["upstream_hashes"], "excluded_people": sorted(excluded),
            "samples": {"regression": regression,
                        "confirmation": select_confirmation(candidates(root), excluded)}}


def validate_plan(root: Path, model_root: Path, plan: dict, commit: str) -> None:
    # Reconstruct deterministic source selection; a hash alone cannot establish independence.
    if plan != make_plan(root, model_root, commit):
        raise ContractError("V2 frozen plan, source evidence, or sample independence changed.")


def environment(root: Path, model_root: Path):
    config = old.load_config(root)
    thresholds = yaml.safe_load((root / old.BASELINE).read_text())["m2"]["fidelity"]
    counter = old.LocalSummaryTokenValidator(model_root / "Qwen3-Embedding-0.6B", maximum_tokens=8192)
    return config, thresholds, counter


def verify(root: Path, model_root: Path, job: str, phase: str, commit: str, *, final: bool = True) -> dict:
    if not job.isdigit() or phase not in PHASES:
        raise ContractError("Invalid v2 job/phase identity.")
    if final:
        old._sentinel(root, job, "COMPLETE", commit)
    directory = root / PRIVATE / job
    report = old.read(root / PUBLIC / f"{phase}-{job}.json")
    required = {"version": generation.VERSION, "source_commit": commit, "job_id": job,
                "phase": phase, "status": "PASS", "contract_hash": stable_hash(contract(root)),
                "test_data_used": False, "embedding_model_run": False, "full_run_authorized": False,
                "scientific_M2_complete": False, "manual_fidelity_review_complete": False}
    if any(report.get(k) != v for k, v in required.items()):
        raise ContractError("V2 report is incomplete or stale.")
    old.verify_manifest(directory, report["private_manifest_hash"])
    plan = old.read(directory / "plan.json")
    validate_plan(root, model_root, plan, commit)
    if report.get("plan_hash") != stable_hash(plan):
        raise ContractError("Report is not bound to the frozen plan.")
    if phase != "regression":
        parent = old.read(directory / "parent.json")
        previous = "regression" if phase == "confirmation" else "confirmation"
        parent_report = verify(root, model_root, parent["job"], previous, commit)
        if (stable_hash(parent_report) != parent["report_hash"]
                or old.read(root / PRIVATE / parent["job"] / "plan.json") != plan):
            raise ContractError("V2 parent evidence changed.")
    if phase == "readiness":
        if report.get("next_action") != "REVIEW_ENGINEERING_BEFORE_PRODUCTION_INTEGRATION":
            raise ContractError("Unexpected production authorization.")
        return report
    config, thresholds, counter = environment(root, model_root)
    for index, source in enumerate(generation.probes()):
        generation.verify_item(directory / f"synthetic-{index}", {"source": source, "condition": "synthetic"},
                               config, thresholds, counter, plan["served_identity"], plan["tokenizer_hash"])
    items = plan["samples"][phase]
    if report.get("checked") != len(items):
        raise ContractError("Incomplete engineering coverage.")
    for i, item in enumerate(items):
        generation.verify_item(directory / f"item-{i:02d}", item, config, thresholds, counter,
                               plan["served_identity"], plan["tokenizer_hash"])
    return report


def run(root: Path, model_root: Path, job: str, phase: str, commit: str, parent: str | None = None) -> dict:
    if phase not in PHASES or not job.isdigit():
        raise ContractError("Invalid v2 phase/job.")
    started = time.monotonic()
    if phase == "regression":
        plan = make_plan(root, model_root, commit)
        parent_record = None
    else:
        if parent is None:
            raise ContractError("V2 successor requires a verified parent.")
        previous = "regression" if phase == "confirmation" else "confirmation"
        verified = verify(root, model_root, parent, previous, commit)
        plan = old.read(root / PRIVATE / parent / "plan.json")
        parent_record = {"job": parent, "report_hash": stable_hash(verified)}
    directory = root / PRIVATE / job
    directory.mkdir(parents=True, exist_ok=False)
    old.write_once(directory / "plan.json", plan)  # Frozen BEFORE any new generation.
    if parent_record:
        old.write_once(directory / "parent.json", parent_record)
    results, probe_results, stopped = [], [], "none"
    if phase != "readiness":
        config, thresholds, counter = environment(root, model_root)
        client = old.PrivateClient(config)
        if client.identity() != plan["served_identity"]:
            raise ContractError("Instruction model metadata differs from the authorized server.")
        for i, source in enumerate(generation.probes()):
            probe_results.append(generation.assess({"source": source, "condition": "synthetic"}, config,
                thresholds, counter, directory / f"synthetic-{i}", plan["served_identity"],
                plan["tokenizer_hash"], client))
            if probe_results[-1]["status"] != "PASS":
                stopped = "synthetic_failure"
                break
        if stopped == "none":
            items = plan["samples"][phase]
            deadline = contract(root)["execution"][f"{phase}_deadline_seconds"]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for start in range(0, len(items), 4):
                    if time.monotonic() - started > deadline - 1800:
                        stopped = "deadline_reserve"
                        break
                    futures = [pool.submit(generation.assess, item, config, thresholds, counter,
                        directory / f"item-{i:02d}", plan["served_identity"], plan["tokenizer_hash"])
                        for i, item in list(enumerate(items))[start:start + 4]]
                    wave = [future.result() for future in futures]
                    results.extend(wave)
                    print(json.dumps({"event": "m2_v2_progress", "phase": phase,
                        "checked": len(results), "planned": len(items),
                        "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
                    if all(r["status"] == "FAIL" for r in wave):
                        stopped = "failure_circuit_breaker"
                        break
        if client.identity() != plan["served_identity"]:
            raise ContractError("Instruction model metadata changed during run.")
    validate_plan(root, model_root, plan, commit)
    passed = (phase == "readiness" or (len(results) == len(plan["samples"][phase])
              and all(r["status"] == "PASS" for r in [*probe_results, *results])))
    old.write_once(directory / "manifest.json", {
        str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.json"))})
    failed = Counter(error for r in results if r["status"] == "FAIL" for error in r["errors"])
    report = {"version": generation.VERSION, "source_commit": commit, "job_id": job, "phase": phase,
              "status": "PASS" if passed else "FAIL", "contract_hash": stable_hash(contract(root)),
              "plan_hash": stable_hash(plan), "private_manifest_hash": file_sha256(directory / "manifest.json"),
              "checked": len(results), "stop_reason": stopped,
              "failure_categories_final_failed_inputs": {k: n if n >= 20 else "<20" for k, n in failed.items()},
              "runtime_seconds": time.monotonic() - started, "test_data_used": False,
              "embedding_model_run": False, "full_run_authorized": False, "scientific_M2_complete": False,
              "manual_fidelity_review_complete": False}
    valid = [r["validation"] for r in results if r["status"] == "PASS"]
    if len(valid) >= 20:
        report["feasibility"] = {"maximum_embedding_tokens": max(r["embedding_tokens"] for r in valid),
            "minimum_retention": {k: min(r["retention"][k] for r in valid) for k in valid[0]["retention"]}}
    if phase == "readiness":
        report["next_action"] = "REVIEW_ENGINEERING_BEFORE_PRODUCTION_INTEGRATION"
        report["remaining_stages"] = ["version_specific_production_worker_and_cache_integration",
            "bounded_train_validation_generation", "0.6B_embedding_and_frozen_40_history_review_packet",
            "human_fidelity_review", "validation_only_logistic_regression"]
    assert_privacy_safe_report(json.dumps(report))
    old.write_once(root / PUBLIC / f"{phase}-{job}.json", report)
    if passed:
        verify(root, model_root, job, phase, commit, final=False)
    return report
