"""Synthetic contracts for retention-bounded selection, never real model tests."""
import copy
import itertools
import json
from pathlib import Path

import pytest
import yaml

from temporal_ehr.final_m23 import selection_v4_production as production
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_selection_v3 as prior
from temporal_ehr.final_m23 import structured_selection_v4 as selection
from temporal_ehr.final_m23.structured_summary import SummaryContractError, synthetic_source

ROOT = Path(__file__).resolve().parents[1]
CONFIG = old.load_config(ROOT)
THRESHOLDS = yaml.safe_load((ROOT / old.BASELINE).read_text())["m2"]["fidelity"]


def source():
    value = synthetic_source()
    for i in range(20):
        value["windows"][0]["facts"].append({
            "code": f"SYN/CODE_ONLY_{i}", "source": "condition", "unit": None,
            "sequence": [{"value": None, "text": None, "count": 1}],
        })
    return value


def test_every_admissible_omission_combination_preserves_all_floors():
    original = source()
    limits = selection.omission_limits(original)
    assert limits["w0"]["max_omissions"] == 2
    assert 0 not in limits["w0"]["omittable_indices"]
    for size in range(3):
        for indices in itertools.combinations(limits["w0"]["omittable_indices"], size):
            _, result, _ = selection.validate(
                {"w0": list(indices), "w1": []}, original,
                thresholds=THRESHOLDS, token_counter=lambda _: 100,
            )
            assert result["retention"]["concepts"] >= .9
            assert all(v == 1 for k, v in result["retention"].items() if k != "concepts")
    assert original == source()


@pytest.mark.parametrize("value,error", [
    ({"w0": [0], "w1": []}, "protected"),
    ({"w0": [1, 2, 3], "w1": []}, "omission_limit"),
    ({"w0": [1, 1], "w1": []}, "duplicate"),
    ({"w0": [2, 1], "w1": []}, "order"),
    ({"w0": [True], "w1": []}, "type"),
    ({"w0": [999], "w1": []}, "invalid"),
    ({"w0": []}, "coverage"),
])
def test_invalid_omission_never_gets_repaired(value, error):
    with pytest.raises(SummaryContractError, match=error):
        selection.materialize(value, source())


def test_protected_features_and_small_windows_and_empty_source():
    original = synthetic_source()
    assert selection.omission_limits(original)["w0"]["max_omissions"] == 0
    assert selection.materialize({"w0": [], "w1": []}, original) == original
    assert selection.materialize({}, {"windows": []}) == {"windows": []}
    assert selection.response_schema(original)["properties"]["w0"]["maxItems"] == 0
    item = {"source": original, "condition": "C1"}
    assert selection.identity(item, CONFIG, "served") != prior.identity(item, CONFIG, "served")
    request = selection.payload(original, CONFIG)
    assert request["temperature"] == 0
    assert request["chat_template_kwargs"]["enable_thinking"] is False


def test_token_overflow_remains_a_rejection():
    with pytest.raises(SummaryContractError, match="embedding_token_budget"):
        selection.validate(selection.full_selection(source()), source(),
                           thresholds=THRESHOLDS, token_counter=lambda _: 8193)


class Client:
    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def request(self, route, request):
        self.calls += 1
        windows = json.loads(request["messages"][1]["content"])["windows"]
        response = {w["key"]: [] for w in windows}
        if self.invalid:
            response["w0"] = [0]
        return {"choices": [{"finish_reason": "stop",
                             "message": {"content": json.dumps(response)}}]}


def test_response_resume_hash_revalidation_and_no_overwrite(tmp_path):
    item = {"source": source(), "condition": "C1"}
    config = CONFIG
    plan = {"served_identity": "served", "tokenizer_hash": "tokenizer"}
    key = selection.identity(item, config, "served")
    env = (config, THRESHOLDS, lambda _: 100)
    client = Client()
    result = production.generate_row(tmp_path, plan, key, item, env, client)
    assert result["status"] == "PASS"
    assert production.generate_row(tmp_path, plan, key, item, env, client) == result
    assert client.calls == 1
    response_path = tmp_path / production.PRIVATE / "generations" / key / "response-1.json"
    response_path.write_text("{}")
    with pytest.raises(old.ContractError, match="hash"):
        production.verified_row(tmp_path, plan, key, item, env)


def test_failed_assessment_is_not_blindly_retried_or_filled(tmp_path):
    item = {"source": source(), "condition": "C1"}
    plan = {"served_identity": "served", "tokenizer_hash": "tokenizer"}
    key = selection.identity(item, CONFIG, "served")
    env = (CONFIG, THRESHOLDS, lambda _: 100)
    client = Client(invalid=True)
    result = production.generate_row(tmp_path, plan, key, item, env, client)
    assert result["status"] == "FAIL" and "summary" not in result
    saved = copy.deepcopy(result)
    assert production.generate_row(tmp_path, plan, key, item, env, client) == saved
    assert client.calls == 2


def test_gate_requires_every_result_to_pass_even_without_error_string(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "load_plan", lambda *_: {})
    monkeypatch.setattr(production, "planned_file", lambda *_: {"items": {"a": {}}})
    monkeypatch.setattr(production.engineering, "environment", lambda *_: (None, None, None))
    monkeypatch.setattr(production, "_run_items", lambda *a, **k: ([{"status": "FAIL", "errors": []}], 1))
    with pytest.raises(old.ContractError, match="rejected"):
        production.run_gate(tmp_path, tmp_path, "commit", "canary", "new")
    report = old.read(tmp_path / production.ENGINEERING_PUBLIC / "canary-new.json")
    assert report["status"] == "FAIL" and report["inputs_accepted"] == 0


def test_new_namespace_and_downstream_guards():
    assert production.PRIVATE != production.prior.PRIVATE
    launcher = (ROOT / "scripts/slurm/submit_m2_selection_v4.sh").read_text()
    for dependency in ("CANARY", "CONFIRMATION", "READINESS", "ARRAY", "SEAL"):
        assert "afterok:$" + dependency in launcher
    assert "--kill-on-invalid-dep=yes" in launcher
    assert 'MODE" = finalize' in launcher
    assert "failed v4 gate requires evidence review" in launcher
    assert 'export M2_SELECTION_READINESS_JOB="$READINESS"' in launcher


def test_shard_cannot_generate_without_semantic_readiness(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "load_plan", lambda *args: {"shards": 1})
    def reject(*args):
        raise old.ContractError("readiness not verified")
    monkeypatch.setattr(production, "verify_readiness", reject)
    with pytest.raises(old.ContractError, match="readiness not verified"):
        production.run_shard(tmp_path, tmp_path, "commit", 0, "job", "missing")
    assert not (tmp_path / production.PRIVATE / "generations").exists()


def test_full_source_feasibility_stops_before_any_generation(tmp_path):
    item = {"source": source()}
    with pytest.raises(old.ContractError, match="no generation authorized"):
        production.check_source_feasibility(tmp_path, {"one": item}, lambda _: 8193)
    report = old.read(tmp_path / production.ENGINEERING_PUBLIC / "source-feasibility.json")
    assert report["status"] == "FAIL" and report["over_budget_inputs"] == "<20"
    assert report["new_generations_requested"] is False
    assert not (tmp_path / production.PRIVATE / "generations").exists()
    assert production.check_source_feasibility(tmp_path / "other", {"one": item}, lambda _: 8192)["status"] == "PASS"


def test_prior_failure_preserved_and_confirmation_must_be_unexposed(tmp_path, monkeypatch):
    monkeypatch.setattr(production, "prior_plan", lambda _: {"verified": True})
    monkeypatch.setattr(production.old, "_sentinel", lambda *args: None)
    monkeypatch.setattr(production.prior.generation, "payload", lambda *args: {"request": "original"})
    gate = {f"old-{i}": {"source": source()} for i in range(32)}
    confirmation = {f"unexposed-{i}": {"source": source()} for i in range(32)}
    monkeypatch.setattr(production.prior, "planned_file", lambda root, plan, name: {
        "items": confirmation if "confirmation" in name else gate
    })
    failed = tmp_path / "outputs/private/run_status/1551/FAILED.json"
    old.write_once(failed, {"state": "FAILED", "evidence": "unchanged"})
    original_bytes = failed.read_bytes()
    old.write_once(tmp_path / production.prior.ENGINEERING_PUBLIC / "canary-1551.json", {
        "status": "FAIL", "inputs_checked": 32, "inputs_accepted": 14,
        "source_commit": production.SOURCE_COMMIT, "plan_hash": production.SOURCE_PLAN_HASH,
        "test_data_used": False, "scientific_M2_complete": False,
        "failure_categories": {"retention_concepts": 23, "identical_rejected_selection": "<20"},
    })
    for i, (key, item) in enumerate(gate.items()):
        directory = tmp_path / production.prior.PRIVATE / "generations" / key
        old.write_once(directory / "source.json", item)
        old.write_once(directory / "request.json", {"request": "original"})
        errors = [] if i < 14 else ["retention_concepts"] * (2 if i < 19 else 1)
        old.write_once(directory / "assessment.json", {"status": "PASS" if i < 14 else "FAIL", "errors": errors})
    evidence = production.source_failure_evidence(tmp_path, {})
    assert len(evidence["gates"]["canary"]) == 32
    assert failed.read_bytes() == original_bytes
    (tmp_path / production.prior.PRIVATE / "generations/unexposed-0").mkdir()
    with pytest.raises(old.ContractError, match="confirmation was exposed"):
        production.source_failure_evidence(tmp_path, {})
    assert failed.read_bytes() == original_bytes
