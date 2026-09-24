"""Synthetic-only regressions for the separately versioned ordered M2 contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.final_m23 import structured_recovery as recovery
from temporal_ehr.final_m23 import structured_summary as structured
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / recovery.CONFIG).read_text())
THRESHOLDS = yaml.safe_load((ROOT / recovery.BASELINE).read_text())["m2"]["fidelity"]


def validate(candidate, source=None):
    return structured.validate(candidate, source or structured.synthetic_source(),
                               thresholds=THRESHOLDS, token_counter=lambda _: 100)


def test_configuration_is_separate_and_legacy_unchanged():
    assert recovery.load_config(ROOT) == CONFIG
    assert file_sha256(ROOT / recovery.BASELINE) == recovery.BASELINE_HASH
    assert CONFIG["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert CONFIG["test_data_authorized"] is False
    assert CONFIG["full_run_authorized"] is False


def test_source_return_to_baseline_survives_and_is_not_increasing():
    source = structured.synthetic_source()
    series = source["windows"][0]["facts"][0]
    assert [r["value"] for r in series["sequence"]] == ["1", "2", "1"]
    assert structured.trajectory(series) == {"net_change": "unchanged", "movement": "fluctuating"}
    assert validate(source)["status"] == "PASS"


def test_historical_canary_exclusion_hash_reconstruction():
    from temporal_ehr.final_m23.m2 import m2_source_text

    text = ('I\n[TIME=1_30_DAYS_BEFORE_T]\n'
            '- SYN/A | value=1 | text_value=negative | unit=u | source=m\n'
            '- SYN/A | value=2 | unit=u | source=m\n'
            '- SYN/B | source=p\n- SYN/A | value=1 | text_value=negative | unit=u | source=m')
    source = structured.ordered_source(text, "I")
    assert recovery.legacy_text_for_exclusion(source) == m2_source_text(text, "I")
    assert recovery.legacy_text_for_exclusion({"windows": []}) == m2_source_text("I", "I")


def test_ordered_runs_preserve_counts_and_separate_units_sources():
    source = structured.ordered_source(
        "I\n[TIME=1_30_DAYS_BEFORE_T]\n"
        "- SYN/A | value=1 | unit=u | source=m\n"
        "- SYN/A | value=1 | unit=u | source=m\n"
        "- SYN/A | value=2 | unit=u | source=m\n"
        "- SYN/A | value=1 | unit=u | source=m\n"
        "- SYN/A | value=1 | unit=v | source=m\n"
        "- SYN/A | value=1 | unit=u | source=p", "I")
    facts = source["windows"][0]["facts"]
    assert len(facts) == 3
    assert [r["count"] for r in facts[0]["sequence"]] == [2, 1, 1]
    assert [r["value"] for r in facts[0]["sequence"]] == ["1", "2", "1"]
    assert validate(source, source)["retention"]["recurrence"] == 1


@pytest.mark.parametrize("replacement", ["SYN/WRONG", "negative"])
def test_text_array_cannot_mask_wrong_code(replacement):
    source = structured.synthetic_source()
    candidate = copy.deepcopy(source)
    candidate["windows"][1]["facts"][0]["code"] = replacement
    with pytest.raises(structured.SummaryContractError):
        validate(candidate)


@pytest.mark.parametrize("mutation", ["count", "order", "value", "unit", "source", "text", "extra_trend"])
def test_corrupt_facts_always_fail(mutation):
    candidate = structured.synthetic_source()
    fact = candidate["windows"][0]["facts"][0]
    if mutation == "count":
        fact["sequence"][0]["count"] = 4
    elif mutation == "order":
        fact["sequence"][0], fact["sequence"][1] = fact["sequence"][1], fact["sequence"][0]
    elif mutation == "value":
        fact["sequence"][0]["value"] = "99"
    elif mutation in {"source", "unit"}:
        fact[mutation] = "invented"
    elif mutation == "text":
        candidate["windows"][1]["facts"][0]["sequence"][0]["text"] = "invented"
    else:
        fact["trend"] = "increasing"
    with pytest.raises(structured.SummaryContractError):
        validate(candidate)


def test_fidelity_checks_are_code_and_window_specific():
    source = structured.ordered_source(
        "I\n[TIME=31_60_DAYS_BEFORE_T]\n- SYN/A | value=1\n- SYN/A | value=2\n"
        "- SYN/B | value=4\n- SYN/B | value=3\n"
        "[TIME=1_30_DAYS_BEFORE_T]\n- SYN/C | value=7", "I")
    assert validate(source, source)["retention"]["trends"] == 1
    candidate = copy.deepcopy(source)
    a, b = candidate["windows"][0]["facts"]
    a["sequence"], b["sequence"] = b["sequence"], a["sequence"]
    with pytest.raises(structured.SummaryContractError):
        validate(candidate, source)
    candidate = copy.deepcopy(source)
    candidate["windows"][1]["facts"].append(candidate["windows"][0]["facts"].pop())
    with pytest.raises(structured.SummaryContractError):
        validate(candidate, source)


def test_falsified_time_labels_never_sorted():
    source = structured.synthetic_source()
    source["windows"][0]["label"] = "[TIME=91_120_DAYS_BEFORE_T]"
    assert validate(source, source)["time_sequence_exact"] is True
    candidate = copy.deepcopy(source)
    candidate["windows"].reverse()
    with pytest.raises(structured.SummaryContractError, match="time_sequence"):
        validate(candidate, source)


def test_no_regex_interpretation_of_clinical_text():
    source = structured.ordered_source(
        'I\n[TIME=1_30_DAYS_BEFORE_T]\n- SYN/A | text_value=["stable"] occurrences=1 | source=m', "I")
    assert set(structured.features(source)["concepts"]) == {("[TIME=1_30_DAYS_BEFORE_T]", "SYN/A", "m", None)}
    assert not structured.features(source)["trends"]
    assert validate(source, source)["status"] == "PASS"


@pytest.mark.parametrize("text", ['{"windows":[],"windows":[]}', '{"windows":NaN}', 'null', '{}'])
def test_strict_json_schema(text):
    with pytest.raises(structured.SummaryContractError):
        structured.check_shape(structured.strict_json(text))


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_invalid_counts(bad):
    source = structured.synthetic_source()
    source["windows"][0]["facts"][0]["sequence"][0]["count"] = bad
    with pytest.raises(structured.SummaryContractError):
        structured.check_shape(source)


def test_numeric_literals_and_missing_values():
    source = structured.ordered_source(
        "I\n[TIME=1_30_DAYS_BEFORE_T]\n- SYN/A | value=1e-5\n- SYN/A | value=-2.0\n- SYN/A", "I")
    assert structured.trajectory(source["windows"][0]["facts"][0]) is None
    assert validate(source, source)["status"] == "PASS"


def test_empty_history_and_token_ceiling():
    source = structured.ordered_source("I", "I")
    assert structured.render(source) == "[NO_PRE_T_EVENTS]"
    assert validate(source, source)["status"] == "PASS"
    with pytest.raises(structured.SummaryContractError, match="embedding_token_budget"):
        structured.validate(source, source, thresholds=THRESHOLDS, token_counter=lambda _: 8193)


def test_retention_is_not_global_value_overlap():
    source = structured.ordered_source("I\n[TIME=1_30_DAYS_BEFORE_T]\n" + "\n".join(
        f"- SYN/{i} | value=1 | unit=u" for i in range(10)), "I")
    candidate = copy.deepcopy(source)
    candidate["windows"][0]["facts"].pop()
    with pytest.raises(structured.SummaryContractError, match="retention_numbers"):
        validate(candidate, source)


def test_generation_key_covers_prompt_schema_source_config_condition_and_identity(monkeypatch):
    source = structured.synthetic_source()
    base = structured.generation_identity(source, "C1", CONFIG, "served")
    assert base != structured.generation_identity(source, "C3", CONFIG, "served")
    assert base != structured.generation_identity(source, "C1", CONFIG, "different")
    altered = copy.deepcopy(CONFIG)
    altered["decoding"]["seed"] += 1
    assert base != structured.generation_identity(source, "C1", altered, "served")
    execution = copy.deepcopy(CONFIG)
    execution["execution"]["concurrency"] = 1
    assert base == structured.generation_identity(source, "C1", execution, "served")
    monkeypatch.setattr(structured, "PROMPT", "changed")
    assert base != structured.generation_identity(source, "C1", CONFIG, "served")


class EchoClient:
    def request(self, path, payload):
        assert path == "chat/completions"
        assert payload["response_format"]["type"] == "json_schema"
        return {"choices": [{"finish_reason": "stop", "message": {"content": payload["messages"][1]["content"]}}]}


def test_assessment_retains_evidence_and_never_overwrites(tmp_path):
    item = {"source": structured.synthetic_source(), "condition": "C1"}
    result = recovery.assess(item, CONFIG, THRESHOLDS, lambda _: 100, tmp_path / "item", "served", EchoClient())
    assert result["status"] == "PASS"
    assert (tmp_path / "item/response-1.json").exists()
    assert result["summary"] == structured.render(item["source"])
    with pytest.raises(FileExistsError):
        recovery.assess(item, CONFIG, THRESHOLDS, len, tmp_path / "item", "served", EchoClient())


def test_rejection_bounded_and_no_fallback(tmp_path, monkeypatch):
    class BadClient(EchoClient):
        calls = 0

        def request(self, path, payload):
            self.calls += 1
            return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}

    monkeypatch.setattr(recovery.time, "sleep", lambda _: None)
    client = BadClient()
    result = recovery.assess({"source": structured.synthetic_source(), "condition": "C1"},
                             CONFIG, THRESHOLDS, len, tmp_path / "item", "served", client)
    assert result["status"] == "FAIL"
    assert "summary" not in result
    assert client.calls == 2
    assert result["errors"] == ["completion_not_stop"] * 2


def test_atomic_write_once_preserves_original(tmp_path):
    path = tmp_path / "FAILED.json"
    recovery.write_once(path, {"status": "FAIL"})
    digest = file_sha256(path)
    with pytest.raises(FileExistsError):
        recovery.write_once(path, {"status": "PASS"})
    assert file_sha256(path) == digest
    assert not list(tmp_path.glob(".pending-*"))


def test_samples_are_training_only_disjoint_across_all_tasks():
    rows = [{"task": task, "condition": condition, "person_hash": str(person),
             "source": {"synthetic": "x" * person}, "split": "train"}
            for task in ("a", "b", "c", "d") for condition in recovery.CONDITIONS for person in range(60)]
    sample = recovery.select_samples(rows, {"5", "15"}, ("a", "b", "c", "d"))
    assert len(sample["development"]) == len(sample["confirmation"]) == 32
    first = {r["person_hash"] for r in sample["development"]} | {"5", "15"}
    second = {r["person_hash"] for r in sample["confirmation"]}
    assert not first & second
    assert sample == recovery.select_samples(list(reversed(rows)), {"5", "15"}, ("a", "b", "c", "d"))
    rows[0]["split"] = "test"
    with pytest.raises(ContractError, match="training-only"):
        recovery.select_samples(rows, set(), ("a", "b", "c", "d"))


def test_runtime_never_runs_on_mac(tmp_path):
    with pytest.raises(ContractError, match="DGX"):
        recovery.check_runtime(tmp_path, "123", True)


def test_private_route_rejects_external_and_redirects():
    config = copy.deepcopy(CONFIG)
    config["api_base"] = "https://example.com/v1"
    with pytest.raises(ContractError):
        recovery.PrivateClient(config)
    with pytest.raises(RuntimeError, match="redirect"):
        recovery.NoRedirect().redirect_request(None, None, None, None, None, "https://example.com")


def test_manifest_detects_modified_or_extra_files(tmp_path):
    recovery.write_once(tmp_path / "item.json", {"synthetic": True})
    recovery.write_once(tmp_path / "manifest.json", {"item.json": file_sha256(tmp_path / "item.json")})
    digest = file_sha256(tmp_path / "manifest.json")
    recovery.verify_manifest(tmp_path, digest)
    recovery.write_once(tmp_path / "extra.json", {})
    with pytest.raises(ContractError, match="contents changed"):
        recovery.verify_manifest(tmp_path, digest)


def test_two_stage_mocked_run_and_semantic_verify(tmp_path, monkeypatch):
    for path in (recovery.CONFIG, recovery.BASELINE):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / path).read_text())
    monkeypatch.setattr(recovery, "old_evidence", lambda _: {"old": "hash"})
    monkeypatch.setattr(recovery, "verify_upstream", lambda *args: None)
    monkeypatch.setattr(recovery, "tokenizer_fingerprint", lambda _: "tokenizer")
    monkeypatch.setattr(recovery, "LocalSummaryTokenValidator", lambda *a, **k: lambda _: 100)

    class Client(EchoClient):
        def __init__(self, config):
            pass

        def identity(self):
            return "served"

    monkeypatch.setattr(recovery, "PrivateClient", Client)

    def prepare(root, directory, config, commit, identity, tokenizer_hash):
        samples = {phase: [{"source": structured.synthetic_source(), "condition": condition,
                           "split": "train", "task": task, "person_hash": f"{phase}-{i}"}
                          for task in recovery.TASKS for condition in recovery.CONDITIONS for i in range(4)]
                   for phase in ("development", "confirmation")}
        plan = {"config_hash": stable_hash(config), "source_commit": commit, "served_identity": identity,
                "tokenizer_hash": tokenizer_hash, "evidence": {"old": "hash"}, "upstream_hashes": {},
                "prior_canary_people": [], "samples": samples}
        recovery.write_once(directory / "plan.json", plan)
        return plan

    monkeypatch.setattr(recovery, "prepare", prepare)
    commit = "a" * 40
    for job, phase, parent in (("1", "development", None), ("2", "confirmation", "1")):
        report = recovery.run(tmp_path, tmp_path, job, phase, commit, parent)
        assert report["status"] == "PASS"
        assert report["scientific_M2_complete"] is False
        assert report["full_run_authorized"] is False
        assert "SYN/" not in json.dumps(report)
        recovery.write_once(tmp_path / f"outputs/private/run_status/{job}/COMPLETE.json",
                            {"job_id": job, "state": "COMPLETE", "git_commit": commit, "exit_status": 0})
        assert recovery.verify_phase(tmp_path, job, phase, commit, lambda _: 100) == report
    with pytest.raises(ContractError):
        recovery.verify_phase(tmp_path, "2", "confirmation", "b" * 40, lambda _: 100)
    path = tmp_path / recovery.PRIVATE / "2/item-00/assessment.json"
    path.write_text('{}')
    with pytest.raises(ContractError, match="contents changed"):
        recovery.verify_phase(tmp_path, "2", "confirmation", commit, lambda _: 100)


@pytest.mark.parametrize("failure", ["probe", "patients", "deadline"])
def test_gate_stops_without_launching_production(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(recovery, "load_config", lambda _: CONFIG)
    baseline = tmp_path / recovery.BASELINE
    baseline.parent.mkdir(parents=True)
    baseline.write_text((ROOT / recovery.BASELINE).read_text())
    monkeypatch.setattr(recovery, "LocalSummaryTokenValidator", lambda *a, **k: len)
    monkeypatch.setattr(recovery, "tokenizer_fingerprint", lambda _: "tokenizer")
    monkeypatch.setattr(recovery, "old_evidence", lambda _: {})
    monkeypatch.setattr(recovery, "verify_upstream", lambda *args: None)

    class Client:
        def __init__(self, config):
            pass

        def identity(self):
            return "served"

    monkeypatch.setattr(recovery, "PrivateClient", Client)
    monkeypatch.setattr(recovery, "prepare", lambda *args: {
        "evidence": {}, "samples": {"development": [{}] * 32}})
    calls = []

    def assessment(item, config, thresholds, counter, directory, identity):
        calls.append(directory.name)
        if directory.name == "synthetic" and failure != "probe":
            return {"status": "PASS", "errors": []}
        return {"status": "FAIL", "errors": ["synthetic_failure"]}

    monkeypatch.setattr(recovery, "assess", assessment)
    clock_values = iter([0, 13000, 13001])
    if failure == "deadline":
        monkeypatch.setattr(recovery.time, "monotonic", lambda: next(clock_values))
    report = recovery.run(tmp_path, tmp_path, "1", "development", "a" * 40)
    assert report["status"] == "FAIL" and report["full_run_authorized"] is False
    assert report["checked"] == (4 if failure == "patients" else 0)
    assert len(calls) == report["checked"] + 1
    assert report["stop_reason"] == {
        "probe": "synthetic_schema_or_fidelity_failure",
        "patients": "failure_circuit_breaker",
        "deadline": "deadline_reserve",
    }[failure]


@pytest.mark.parametrize("problem", ["overlap", "test", "duplicate", "prior"])
def test_sample_plan_rejects_contamination(problem):
    plan = {"prior_canary_people": [], "samples": {
        phase: [{"task": task, "condition": condition, "split": "train", "person_hash": f"{phase}-{i}"}
                for task in recovery.TASKS for condition in recovery.CONDITIONS for i in range(4)]
        for phase in ("development", "confirmation")}}
    row = plan["samples"]["confirmation"][0]
    if problem == "overlap":
        row["person_hash"] = "development-0"
    elif problem == "test":
        row["split"] = "test"
    elif problem == "duplicate":
        row["person_hash"] = "confirmation-1"
    else:
        plan["prior_canary_people"] = [row["person_hash"]]
    with pytest.raises(ContractError):
        recovery.verify_samples(plan)


def test_slurm_only_submits_two_engineering_jobs():
    submit = (ROOT / "scripts/slurm/submit_m2_structured_gates.sh").read_text()
    batch = (ROOT / "scripts/slurm/24e_m2_structured_gate.sbatch").read_text()
    assert submit.count("sbatch --parsable") == 2
    assert '--dependency="afterok:${DEVELOPMENT_JOB_ID}"' in submit
    assert "--kill-on-invalid-dep=yes" in submit
    assert "--array" not in submit
    assert "#SBATCH --gres" not in batch
    assert "CONTAINER_IMAGE=" in batch
    assert 'final_container_python "${PYTHON}" scripts/run_m2_structured.py' in batch
    assert "final_mark_success" in batch
