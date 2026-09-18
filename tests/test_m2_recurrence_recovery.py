from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.final_m23 import recurrence_recovery as recovery
from temporal_ehr.final_m23.m2 import (
    NonRetryableSummaryError,
    SummaryGeneration,
    bounded_summary_payload_is_reusable,
    group_structured_facts,
    validate_bounded_summary,
)
from temporal_ehr.pilot.cache import atomic_json, file_sha256
from temporal_ehr.serialization.integrity import stable_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/final_m2_m3.yaml"
M2 = yaml.safe_load(CONFIG.read_text())["m2"]
SOURCE = "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/SYNTHETIC | value=5 | unit=mg | source=measurement"
GROUPED = group_structured_facts(SOURCE)
SUMMARY = "[TIME=1_30_DAYS_BEFORE_T]\n- LOINC/SYNTHETIC | values=5 | unit=mg"


def validate(summary: str, source: str = GROUPED) -> dict:
    return validate_bounded_summary(
        summary, source_text=source, token_counter=lambda text: len(text), thresholds=M2["fidelity"]
    )


def test_singleton_renderer_and_validator_agree():
    assert "occurrences=1" in GROUPED
    implicit = validate(SUMMARY)
    explicit = validate(SUMMARY + " | occurrences=1")
    assert implicit["retention"] == explicit["retention"]
    assert explicit["status"] == "PASS"


@pytest.mark.parametrize("count", ["0", "-1", "1.0", "1e0", "2", "999"])
def test_singleton_fix_does_not_admit_invalid_or_invented_counts(count):
    with pytest.raises(NonRetryableSummaryError):
        validate(SUMMARY + f" | occurrences={count}")


def test_explicit_singleton_cannot_disguise_missing_recurrence():
    with pytest.raises(NonRetryableSummaryError):
        validate(SUMMARY + " | occurrences=1", GROUPED.replace("occurrences=1", "occurrences=2"))


def test_legacy_valid_cache_remains_reusable_without_rewrite():
    validation = validate(SUMMARY)
    payload = {"summary": SUMMARY, "bounded_validation": validation, "bounded_validation_passed": True}
    original = json.dumps(payload, sort_keys=True)
    assert bounded_summary_payload_is_reusable(
        payload, source_text=GROUPED, token_counter=len, thresholds=M2["fidelity"]
    )
    assert original == json.dumps(payload, sort_keys=True)


@pytest.mark.parametrize(
    "summary",
    [
        SUMMARY.replace("values=5", "values=6"),
        SUMMARY.replace("unit=mg", "unit=kg"),
        SUMMARY.replace("1_30", "31_60"),
        "None",
    ],
)
def test_other_fidelity_gates_remain_closed(summary):
    with pytest.raises(NonRetryableSummaryError):
        validate(summary)


def _rows():
    return [
        {
            "split": split,
            "representation": condition.split(":")[0],
            "perturbation_seed": 20260815 if "C3" in condition else None,
            "text": "unused instruction\n" + SOURCE + "\n- LOINC/" + "X" * length,
            "target": length % 2,
        }
        for split in ("train", "validation", "test")
        for condition in recovery.CONDITIONS
        for length in range(1, 12)
    ]


def test_canary_sampling_is_outcome_order_independent_and_train_only():
    rows = _rows()
    selected = recovery.select_canary_sources(rows, "unused instruction")
    for row in rows:
        row["target"] = 1 - row["target"]
        if row["split"] != "train":
            row["text"] = "DO NOT USE VALIDATION OR TEST"
    assert selected == recovery.select_canary_sources(list(reversed(rows)), "unused instruction")
    assert len(selected) == 8
    assert not any("DO NOT USE" in item["source_text"] for item in selected)
    assert all(set(item) == {"condition", "source_text"} for item in selected)


def test_missing_training_stratum_fails():
    with pytest.raises(ContractError):
        recovery.select_canary_sources([], "instruction")


def test_rejected_attempts_are_counted_and_private(tmp_path):
    class FakeClient:
        def summarize_response(self, **kwargs):
            return SummaryGeneration(SUMMARY + " | occurrences=4", "stop", 20, 30)

    result = recovery.assess_source(FakeClient(), {"source_text": GROUPED}, M2, len, tmp_path)
    assert result["status"] == "FAIL"
    assert result["errors"] == {"recurrence_mismatch_or_retention": 3}
    assert json.loads((tmp_path / "assessment.json").read_text()) == result
    assert "LOINC" not in json.dumps(result)


def test_failure_category_never_exports_error_content():
    assert recovery.failure_category(RuntimeError("PRIVATE arbitrary server content")) == "endpoint_or_other"


def _status(root, job, state, commit):
    path = root / "outputs/private/run_status" / job / f"{state}.json"
    atomic_json(
        path, {"job_id": job, "state": state, "git_commit": commit, "exit_status": 0 if state == "COMPLETE" else 1}
    )
    return path


def test_original_sentinels_are_not_written(tmp_path):
    failed = _status(tmp_path, "596_0", "FAILED", recovery.SOURCE_596)
    for job in [*(f"470_{i}" for i in range(4)), *(f"471_{i}" for i in range(12))]:
        _status(tmp_path, job, "COMPLETE", recovery.SOURCE_M3)
    before = failed.read_bytes()
    evidence = recovery.preserved_evidence(tmp_path)
    assert evidence[str(failed.relative_to(tmp_path))] == file_sha256(failed)
    assert failed.read_bytes() == before
    _status(tmp_path, "596_0", "COMPLETE", recovery.SOURCE_596)
    with pytest.raises(ContractError):
        recovery.preserved_evidence(tmp_path)


def _canary_fixture(tmp_path, monkeypatch):
    commit = "a" * 40
    _status(tmp_path, "600", "COMPLETE", commit)
    config = tmp_path / "configs/experiments/final_m2_m3.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("fixture config")
    monkeypatch.setattr(recovery, "preserved_evidence", lambda root: {"old": "unchanged"})
    private = tmp_path / recovery.PRIVATE_ROOT / "600"
    results = [{"status": "PASS", "errors": {}} for _ in range(32)]
    for index, result in enumerate(results):
        atomic_json(private / f"item-{index:02d}/assessment.json", result)
    manifest = {str(path.relative_to(private)): file_sha256(path) for path in sorted(private.rglob("*.json"))}
    atomic_json(private / "manifest.json", manifest)
    report = {
        "schema": recovery.SCHEMA,
        "status": "PASS",
        "job_id": "600",
        "recovery_git_commit": commit,
        "source_git_commit": recovery.SOURCE_596,
        "planned_inputs": 32,
        "all_inputs_passed": True,
        "test_data_used": False,
        "config_sha256": file_sha256(config),
        "original_evidence_hash": stable_hash({"old": "unchanged"}),
        "private_manifest_sha256": file_sha256(private / "manifest.json"),
        "private_assessments_hash": stable_hash(results),
    }
    path = tmp_path / recovery.REPORT_ROOT / "canary-600.json"
    atomic_json(path, report)
    return commit, path, report


def test_canary_semantic_and_hash_gate(tmp_path, monkeypatch):
    commit, path, report = _canary_fixture(tmp_path, monkeypatch)
    assert recovery.verify_canary(tmp_path, "600", commit)["status"] == "PASS"
    report["all_inputs_passed"] = False
    atomic_json(path, report)
    with pytest.raises(ContractError):
        recovery.verify_canary(tmp_path, "600", commit)


def test_canary_tampering_or_stale_commit_rejected(tmp_path, monkeypatch):
    commit, _, _ = _canary_fixture(tmp_path, monkeypatch)
    with pytest.raises(ContractError):
        recovery.verify_canary(tmp_path, "600", "b" * 40)
    atomic_json(tmp_path / recovery.PRIVATE_ROOT / "600/item-00/assessment.json", {"status": "FAIL"})
    with pytest.raises(ContractError):
        recovery.verify_canary(tmp_path, "600", commit)


def test_submitter_cannot_launch_scientific_graph():
    script = (ROOT / "scripts/slurm/submit_m2_after596_canary.sh").read_text()
    assert "24d_m2_after596_canary.sbatch" in script
    assert "24c_m2_summary_shard_array.sbatch" not in script
    assert "24b_m2_embed_task_array.sbatch" not in script
    assert 'QUEUE="$(squeue' in script
    batch = (ROOT / "scripts/slurm/24d_m2_after596_canary.sbatch").read_text()
    assert 'final_container_python "${PYTHON}" scripts/run_m2_after596_canary.py' in batch
    assert batch.index("run_m2_after596_canary.py") < batch.index("final_mark_success")


@pytest.mark.parametrize("bad_output", [False, True])
def test_entire_canary_uses_mock_api_and_writes_fail_closed_report(tmp_path, monkeypatch, bad_output):
    config = tmp_path / "configs/experiments/final_m2_m3.yaml"
    config.parent.mkdir(parents=True)
    config.write_bytes(CONFIG.read_bytes())
    tasks = ("lab_anemia", "guo_readmission", "new_hypertension", "chexpert")
    spec = SimpleNamespace(
        m2=M2,
        tasks=tasks,
        m1_config_path=Path("fixture"),
        model_directory_name="Qwen3-Embedding-0.6B",
        model_repository="Qwen/Qwen3-Embedding-0.6B",
    )
    monkeypatch.setattr(recovery, "require_prompt4_hard_gates", lambda *args: None)
    monkeypatch.setattr(recovery, "preserved_evidence", lambda *args: {"original": "unchanged"})
    monkeypatch.setattr(recovery, "load_final_m23_authorization", lambda *args, **kwargs: spec)
    monkeypatch.setattr(
        recovery,
        "load_final_m1_authorization",
        lambda *args, **kwargs: SimpleNamespace(instructions=dict.fromkeys(tasks, "unused instruction")),
    )
    monkeypatch.setattr(recovery, "read_final_task_texts", lambda *args: _rows())
    monkeypatch.setattr(recovery, "LocalSummaryTokenValidator", lambda *args, **kwargs: len)

    def fake_request(self, path, payload=None):
        if payload is None:
            return {"data": [{"id": M2["instruction_model"]}]}
        summary = re.sub(r" \| source=[^|\n]+", "", payload["messages"][1]["content"])
        if bad_output:
            summary = summary.replace("occurrences=1", "occurrences=999")
        return {"choices": [{"message": {"content": summary}, "finish_reason": "stop"}]}

    monkeypatch.setattr(recovery.OnPremiseSummaryClient, "_request", fake_request)
    report = recovery.run_canary(tmp_path, tmp_path / "models", "600", "a" * 40)
    assert report["status"] == ("FAIL" if bad_output else "PASS")
    assert report["scientific_M2_complete"] is False
    assert report["test_data_used"] is False
    assert "LOINC/" not in json.dumps(report)
    assert len(list((tmp_path / recovery.PRIVATE_ROOT / "600").glob("item-*/assessment.json"))) == 32
    assert (tmp_path / recovery.PRIVATE_ROOT / "600/item-00/response-1.json").is_file()
    _status(tmp_path, "600", "FAILED" if bad_output else "COMPLETE", "a" * 40)
    if bad_output:
        with pytest.raises(ContractError):
            recovery.verify_canary(tmp_path, "600", "a" * 40)
    else:
        assert recovery.verify_canary(tmp_path, "600", "a" * 40)["status"] == "PASS"
