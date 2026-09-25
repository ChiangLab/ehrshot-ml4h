"""Synthetic-only regression for the prompt amendment and fail-closed batch chain."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_v2 as new
from temporal_ehr.final_m23 import structured_v2_recovery as pipeline
from temporal_ehr.final_m23.structured_summary import generation_identity
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = old.load_config(ROOT)
THRESHOLDS = yaml.safe_load((ROOT / old.BASELINE).read_text())["m2"]["fidelity"]


class Client:
    calls = 0

    def __init__(self, config=None, mutate=None):
        self.mutate = mutate

    def identity(self):
        return "served"

    def request(self, route, payload):
        type(self).calls += 1
        assert route == "chat/completions"
        document = json.loads(payload["messages"][1]["content"])
        if self.mutate:
            self.mutate(document)
        return {"id": f"different-response-{self.calls}", "choices": [{"finish_reason": "stop",
            "message": {"content": json.dumps(document)}}]}


def test_prompt_only_amendment_keeps_source_decoding_schema_and_changes_identity():
    source = new.probes()[0]
    request = new.payload(source, CONFIG)
    original = old.request_payload(source, CONFIG)
    assert json.loads(request["messages"][1]["content"]) == json.loads(original["messages"][1]["content"])
    assert request == new.payload(json.loads(json.dumps(source, sort_keys=True)), CONFIG)
    assert request["response_format"]["json_schema"]["schema"] == original["response_format"]["json_schema"]["schema"]
    assert all(request[k] == v for k, v in CONFIG["decoding"].items())
    assert request["messages"][0]["content"] != original["messages"][0]["content"]
    item = {"source": source, "condition": "synthetic"}
    assert new.identity(item, CONFIG, "served") != generation_identity(source, "synthetic", CONFIG, "served")


@pytest.mark.parametrize("mutation", ["duplicate", "counts", "runs", "facts", "unit", "null", "omission"])
def test_bad_outputs_rejected_without_repair(tmp_path, mutation):
    source = new.probes()[0]
    original = copy.deepcopy(source)

    def mutate(document):
        facts = document["windows"][0]["facts"]
        if mutation == "duplicate":
            facts.append(copy.deepcopy(facts[0]))
        elif mutation == "counts":
            facts[0]["sequence"][0]["count"] += 1
        elif mutation == "runs":
            facts[0]["sequence"][0], facts[0]["sequence"][1] = facts[0]["sequence"][1], facts[0]["sequence"][0]
        elif mutation == "facts":
            facts.reverse()
        elif mutation == "unit":
            facts[0]["unit"] = "invented"
        elif mutation == "null":
            facts[0]["sequence"][0]["text"] = "null"
        else:
            facts.clear()

    directory = tmp_path / "row"
    result = new.assess({"source": source, "condition": "synthetic"}, CONFIG, THRESHOLDS,
                        lambda _: 100, directory, "served", "tokenizer", Client(mutate=mutate))
    assert result["status"] == "FAIL"
    assert result["errors"][-1] == "identical_rejected_content"
    assert source == original
    assert "summary" not in result
    assert (directory / "response-2.json").is_file()


@pytest.mark.parametrize("source", new.probes())
def test_valid_facts_roundtrip_and_bind_to_response(tmp_path, source):
    item = {"source": source, "condition": "synthetic"}
    directory = tmp_path / "row"
    result = new.assess(item, CONFIG, THRESHOLDS, lambda _: 100, directory, "served", "tok", Client())
    assert new.verify_item(directory, item, CONFIG, THRESHOLDS, lambda _: 100, "served", "tok") == result
    assert result["document"] == source
    assert result["attempts"] == 1
    with pytest.raises(FileExistsError):
        new.assess(item, CONFIG, THRESHOLDS, lambda _: 100, directory, "served", "tok", Client())
    payload = old.read(directory / "response-1.json")
    payload["choices"][0]["message"]["content"] = '{"windows": []}'
    if source["windows"]:
        (directory / "response-1.json").write_text(json.dumps(payload))
        with pytest.raises(ValueError):
            new.verify_item(directory, item, CONFIG, THRESHOLDS, lambda _: 100, "served", "tok")


def synthetic_candidates():
    return [{"task": task, "condition": condition, "person_hash": f"person-{i}", "split": "train",
             "prepared_text_hash": stable_hash([task, condition, i]), "source": new.probes()[0]}
            for task in old.TASKS for condition in old.CONDITIONS for i in range(20)]


def test_selection_is_deterministic_balanced_and_excludes_people_globally():
    rows = synthetic_candidates()
    excluded = {f"person-{i}" for i in range(8)}
    selected = pipeline.select_confirmation(rows, excluded)
    assert len(selected) == 32
    assert not excluded & {r["person_hash"] for r in selected}
    assert selected == pipeline.select_confirmation(list(reversed(rows)), excluded)
    with pytest.raises(ContractError):
        pipeline.select_confirmation(rows, {f"person-{i}" for i in range(19)})
    with pytest.raises(ContractError):
        pipeline.select_confirmation([{**rows[0], "split": "test"}, *rows[1:]], excluded)
    with pytest.raises(ContractError):
        pipeline.select_confirmation([*rows, rows[0]], set())


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for path in (old.BASELINE, old.CONFIG):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / path).read_text())
    rows = synthetic_candidates()
    regression = [r for r in rows if int(r["person_hash"].split("-")[-1]) < 8]
    original = {"samples": {"development": regression[:32], "confirmation": regression[32:]},
                "prior_canary_people": ["person-8"], "upstream_hashes": {},
                "tokenizer_hash": "tokenizer", "served_identity": "served"}
    monkeypatch.setattr(pipeline, "evidence", lambda *a: (original, {"old_failed_hash": "untouched"}))
    monkeypatch.setattr(pipeline, "candidates", lambda *a: rows)
    monkeypatch.setattr(old, "LocalSummaryTokenValidator", lambda *a, **k: lambda _: 100)
    monkeypatch.setattr(old, "PrivateClient", Client)
    return tmp_path


def complete(root, job):
    old.write_once(root / f"outputs/private/run_status/{job}/COMPLETE.json", {
        "job_id": job, "state": "COMPLETE", "exit_status": 0, "git_commit": "a" * 40})


def test_entire_chain_binds_fresh_plan_and_never_reports_scientific_completion(environment):
    root = environment
    before = Client.calls
    regression = pipeline.run(root, root, "1001", "regression", "a" * 40)
    assert regression["checked"] == 64
    complete(root, "1001")
    confirmation = pipeline.run(root, root, "1002", "confirmation", "a" * 40, "1001")
    assert confirmation["checked"] == 32
    complete(root, "1002")
    calls_before_readiness = Client.calls
    readiness = pipeline.run(root, root, "1003", "readiness", "a" * 40, "1002")
    complete(root, "1003")
    assert Client.calls == calls_before_readiness
    assert calls_before_readiness - before == 102  # 96 clinical fixtures + 6 synthetic probes.
    assert pipeline.verify(root, root, "1003", "readiness", "a" * 40) == readiness
    assert readiness["scientific_M2_complete"] is False
    assert readiness["full_run_authorized"] is False
    assert "SYN/" not in json.dumps(readiness)
    with pytest.raises(FileExistsError):
        pipeline.run(root, root, "1001", "regression", "a" * 40)
    old.write_once(root / "outputs/private/run_status/1001/FAILED.json", {"state": "FAILED"})
    with pytest.raises(ContractError):
        pipeline.verify(root, root, "1003", "readiness", "a" * 40)


def test_failed_regression_blocks_next_phase_and_keeps_response_evidence(environment, monkeypatch):
    root = environment

    class Bad(Client):
        def request(self, route, payload):
            return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}

    monkeypatch.setattr(old, "PrivateClient", Bad)
    report = pipeline.run(root, root, "1001", "regression", "a" * 40)
    assert report["status"] == "FAIL"
    assert report["stop_reason"] == "synthetic_failure"
    assert (root / pipeline.PRIVATE / "1001/synthetic-0/response-1.json").exists()
    with pytest.raises(ContractError):
        pipeline.run(root, root, "1002", "confirmation", "a" * 40, "1001")
    assert not (root / pipeline.PRIVATE / "1002").exists()


@pytest.mark.parametrize("drift", ["plan", "response", "commit"])
def test_chain_rejects_tampering(environment, drift):
    root = environment
    pipeline.run(root, root, "1001", "regression", "a" * 40)
    complete(root, "1001")
    commit = "a" * 40
    if drift == "commit":
        commit = "b" * 40
    else:
        relative = "plan.json" if drift == "plan" else "item-00/response-1.json"
        (root / pipeline.PRIVATE / "1001" / relative).write_text('{}')
    with pytest.raises(ContractError):
        pipeline.verify(root, root, "1001", "regression", commit)


def test_containerized_afterok_chain_is_bounded_and_does_not_launch_legacy_production():
    launcher = (ROOT / "scripts/slurm/submit_m2_structured_v2.sh").read_text()
    batch = (ROOT / "scripts/slurm/24g_m2_structured_v2.sbatch").read_text()
    assert launcher.count("sbatch --parsable") == 3
    assert '--dependency="afterok:${REGRESSION_JOB_ID}"' in launcher
    assert '--dependency="afterok:${CONFIRMATION_JOB_ID}"' in launcher
    assert launcher.count("--kill-on-invalid-dep=yes") == 2
    assert "--array" not in launcher
    assert "noclobber" in launcher
    assert "CONTAINER_IMAGE=" in batch
    assert 'final_container_python "$PYTHON" scripts/run_m2_structured_v2.py' in batch
    assert batch.index("scripts/run_m2_structured_v2.py") < batch.index("final_mark_success")


def test_original_986_failure_and_hashes_must_be_preserved(tmp_path, monkeypatch):
    root = tmp_path
    items = synthetic_candidates()[:32]
    plan = {"samples": {"confirmation": items}, "tokenizer_hash": "tok", "served_identity": "served"}
    old_evidence = {"root_evidence": "fixed"}
    replay = {"status": "PASS", "job_id": "985"}
    monkeypatch.setattr(pipeline.prior, "verify", lambda *a, **k: replay)
    monkeypatch.setattr(pipeline.prior, "source_evidence", lambda *a: (plan, old_evidence))
    monkeypatch.setattr(old, "load_config", lambda *a: CONFIG)
    directory = root / pipeline.prior.PRIVATE / "986"
    old.write_once(directory / "plan.json", plan)
    old.write_once(directory / "evidence.json", old_evidence)
    old.write_once(directory / "parent.json", {"job": "985", "report_hash": stable_hash(replay)})
    for i, item in enumerate(items):
        old.write_once(directory / f"item-{i:02d}/source.json", item)
        old.write_once(directory / f"item-{i:02d}/admission.json", {"status": "PASS" if i < 29 else "FAIL"})
    old.write_once(directory / "manifest.json", {
        str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.json"))})
    digest = file_sha256(directory / "manifest.json")
    monkeypatch.setattr(pipeline, "PRIOR_MANIFEST", digest)
    old.write_once(root / pipeline.prior.PUBLIC / "confirmation-986.json", {
        "status": "FAIL", "checked": 32, "job_id": "986", "phase": "confirmation",
        "private_manifest_hash": digest, "recovery_commit": pipeline.PRIOR_COMMIT,
        "test_data_used": False, "embedding_model_run": False, "generation_config_hash": stable_hash(CONFIG),
        "evidence_hash": stable_hash(old_evidence), "tokenizer_hash": "tok", "served_identity": "served"})
    failure = root / "outputs/private/run_status/986/FAILED.json"
    old.write_once(failure, {"job_id": "986", "state": "FAILED", "exit_status": 1,
                             "git_commit": pipeline.PRIOR_COMMIT})
    before = file_sha256(failure)
    assert pipeline.evidence(root, root)[0] == plan
    assert file_sha256(failure) == before
    (directory / "item-00/source.json").write_text('{}')
    with pytest.raises(ContractError, match="contents changed"):
        pipeline.evidence(root, root)
    assert file_sha256(failure) == before


def test_overbudget_and_truncation_never_admitted(tmp_path):
    item = {"source": new.probes()[0], "condition": "synthetic"}
    result = new.assess(item, CONFIG, THRESHOLDS, lambda _: 8193,
                        tmp_path / "budget", "served", "tok", Client())
    assert result["status"] == "FAIL"
    assert result["errors"][0] == "embedding_token_budget"

    class Truncated(Client):
        def request(self, route, payload):
            response = super().request(route, payload)
            response["choices"][0]["finish_reason"] = "length"
            return response

    result = new.assess(item, CONFIG, THRESHOLDS, lambda _: 100,
                        tmp_path / "truncated", "served", "tok", Truncated())
    assert result["status"] == "FAIL"
    assert result["errors"] == ["completion_not_stop", "completion_not_stop"]
