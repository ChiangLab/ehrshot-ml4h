"""Synthetic production integration only: no model, governed source, or endpoint."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.final_m23 import structured_production as p
from temporal_ehr.final_m23 import structured_reference as r
from temporal_ehr.final_m23.m2 import MANUAL_REVIEW_CRITERIA
from temporal_ehr.pilot.cache import atomic_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = p.old.load_config(ROOT)
THRESHOLDS = yaml.safe_load((ROOT / p.old.BASELINE).read_text())["m2"]["fidelity"]
ENV = (CONFIG, THRESHOLDS, lambda _: 100)
COMMIT = "a" * 40


class Client:
    calls = 0

    def __init__(self, config=None):
        pass

    def identity(self):
        return "served"

    def request(self, route, payload):
        Client.calls += 1
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": payload["messages"][1]["content"]}}]}


def item():
    return {"source": p.generation.probes()[0], "condition": p.C1}


def minimal_plan():
    return {"served_identity": "served", "tokenizer_hash": "tok"}


def test_generation_reuses_and_detects_tampering(tmp_path):
    plan, sample = minimal_plan(), item()
    key = p.generation.identity(sample, CONFIG, "served")
    start = Client.calls
    first = p.generate_row(tmp_path, plan, key, sample, ENV, Client())
    assert p.generate_row(tmp_path, plan, key, sample, ENV, Client()) == first
    assert Client.calls - start == 1
    path = tmp_path / p.PRIVATE / "generations" / key / "response-1.json"
    atomic_json(path, {"changed": True})
    with pytest.raises(ContractError, match="hash"):
        p.generate_row(tmp_path, plan, key, sample, ENV, Client())


def test_saved_response_after_interruption_is_not_regenerated(tmp_path):
    plan, sample = minimal_plan(), item()
    key = p.generation.identity(sample, CONFIG, "served")
    directory = tmp_path / p.PRIVATE / "generations" / key
    response = Client().request("chat/completions", p.generation.payload(sample["source"], CONFIG))
    p.persist(directory / "response-1.json", response)
    start = Client.calls
    assert p.generate_row(tmp_path, plan, key, sample, ENV, Client())["status"] == "PASS"
    assert Client.calls == start


def test_interrupted_call_consumes_budget_without_infinite_retries(tmp_path):
    plan, sample = minimal_plan(), item()
    key = p.generation.identity(sample, CONFIG, "served")
    directory = tmp_path / p.PRIVATE / "generations" / key
    for attempt in (1, 2):
        p.persist(directory / f"started-{attempt}.json", {"attempt": attempt})
    start = Client.calls
    assert p.generate_row(tmp_path, plan, key, sample, ENV, Client())["status"] == "FAIL"
    assert p.generate_row(tmp_path, plan, key, sample, ENV, Client())["status"] == "FAIL"
    assert Client.calls == start


def test_rejected_rows_never_enter_cache_and_budget_never_resets(tmp_path):
    class Bad(Client):
        def request(self, *args):
            Client.calls += 1
            return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}
    plan, sample = minimal_plan(), item()
    key = p.generation.identity(sample, CONFIG, "served")
    start = Client.calls
    result = p.generate_row(tmp_path, plan, key, sample, ENV, Bad())
    assert result["status"] == "FAIL"
    assert p.generate_row(tmp_path, plan, key, sample, ENV, Bad()) == result
    assert Client.calls - start == 2
    assert not (tmp_path / p.PRIVATE / "generations" / key / "SEALED.json").exists()


def test_shards_are_bounded_unique_and_deterministic():
    values = {str(i): {} for i in range(79)}
    parts = p.split_work(values)
    assert parts == p.split_work(dict(reversed(list(values.items()))))
    assert max(map(len, parts)) == 16
    assert len([k for part in parts for k in part]) == len(values)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    for file in (p.old.CONFIG, p.old.BASELINE):
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / file).read_text())
    ready = {"plan_hash": p.ENGINEERING_PLAN, "private_manifest_hash": p.READINESS_MANIFEST}
    monkeypatch.setattr(p.engineering, "verify", lambda *a: ready)
    p.persist(tmp_path / p.engineering.PRIVATE / "990/plan.json", {
        "served_identity": "served", "tokenizer_hash": "tok", "samples": {"regression": [], "confirmation": []}})
    monkeypatch.setattr(p.engineering, "environment", lambda *a: ENV)
    monkeypatch.setattr(p.old, "PrivateClient", Client)
    monkeypatch.setattr(p.old, "tokenizer_fingerprint", lambda *a: "tok")
    identity = SimpleNamespace(artifact_hash="model", tokenizer_hash="tok")
    monkeypatch.setattr(p, "load_model_identity", lambda *a, **k: identity)
    monkeypatch.setattr(r, "load_model_identity", lambda *a, **k: identity)
    rows = {}
    for task in p.old.TASKS:
        rows[task] = []
        for condition in (p.C1, "C3_shuffled_earlier_labels:seed=20260815"):
            for split in ("train", "validation"):
                if condition != p.C1 and split == "train":
                    continue
                for i in range(40):
                    source = copy.deepcopy(item()["source"])
                    source["windows"][0]["facts"][0]["sequence"][0]["value"] = str(i + 100)
                    rows[task].append({"task": task, "condition": condition, "split": split,
                        "instance_key": f"synthetic-{split}-{i}", "target": bool(i % 2),
                        "source_text": f"Synthetic source {i}", "item": {"source": source, "condition": condition}})
    monkeypatch.setattr(p, "source_rows", lambda *a: iter(copy.deepcopy(list(rows.items()))))
    class Encoder:
        dimension = 1024
        calls = 0
        def __init__(self, *a, **k):
            Encoder.calls += 1
        def encode(self, texts, **k):
            result = np.zeros((len(texts), 1024), dtype=np.float32)
            result[:, 0] = 1
            return result
    monkeypatch.setattr(r, "LocalQwenEncoder", Encoder)
    spec = SimpleNamespace(evaluation={"regularization_grid": [0.1], "logistic_seed": 20260815})
    monkeypatch.setattr(p.old, "load_final_m23_authorization", lambda *a, **k: spec)
    p.freeze_plan(tmp_path, tmp_path, COMMIT)
    return tmp_path, Encoder


def complete(root, job):
    p.persist(root / "outputs/private/run_status" / job / "COMPLETE.json", {
        "state": "COMPLETE", "job_id": job, "git_commit": COMMIT, "exit_status": 0})


def test_entire_reference_chain_and_read_only_semantic_verifier(pipeline):
    root, encoder = pipeline
    plan = p.load_plan(root, COMMIT)
    for index in range(plan["shards"]):
        p.run_shard(root, root, COMMIT, index, f"100_{index}")
        complete(root, f"100_{index}")
    report = r.seal_generation(root, root, COMMIT, "100")
    assert report["unique_inputs"] == plan["unique_inputs"]
    assert report["scientific_M2_complete"] is False
    review = root / p.PRIVATE / "review"
    assert len(p.old.read(review / "packet.json")["entries"]) == 40
    with pytest.raises(ContractError, match="incomplete"):
        r.fit(root, root, COMMIT, "200")
    assert not (root / p.PRIVATE / "fit").exists()
    for index, task in enumerate(p.old.TASKS):
        r.embed(root, root, COMMIT, task, f"200_{index}")
        complete(root, f"200_{index}")
        calls = encoder.calls
        r.embed(root, root, COMMIT, task, f"201_{index}")
        assert encoder.calls == calls
    decisions = p.old.read(review / "decisions.json")
    decisions["review_complete"] = True
    for decision in decisions["decisions"]:
        decision.update(dict.fromkeys(MANUAL_REVIEW_CRITERIA, True))
    atomic_json(review / "decisions.json", decisions)
    fitted = r.fit(root, root, COMMIT, "200")
    complete(root, "300")
    assert fitted["reference_complete"] is True
    assert fitted["test_data_used"] is False
    assert fitted["sensitivity_complete"] is False
    before = {str(f): p.file_sha256(f) for f in (root / p.PRIVATE).rglob("*") if f.is_file()}
    assert r.verify_reference(root, COMMIT, "200", "300")["status"] == "PASS"
    after = {str(f): p.file_sha256(f) for f in (root / p.PRIVATE).rglob("*") if f.is_file()}
    assert after == before
    for path in (root / p.PUBLIC).rglob("*.json"):
        text = path.read_text()
        assert "Synthetic source" not in text and "synthetic-train-" not in text
        assert '"source_text"' not in text and '"predictions"' not in text
    p.persist(root / "outputs/private/run_status/300/FAILED.json", {"state": "FAILED"})
    with pytest.raises(ContractError, match="conflicting"):
        r.verify_reference(root, COMMIT, "200", "300")


def test_plan_drift_and_path_escape_fail_closed(pipeline):
    root, _ = pipeline
    with pytest.raises(ContractError):
        p.load_plan(root, "b" * 40)
    with pytest.raises(ContractError):
        p.checked_file(root, "../../escape", "x")
    plan = p.load_plan(root, COMMIT)
    path = root / p.PRIVATE / "plan/shard-0.json"
    atomic_json(path, {"items": {}})
    with pytest.raises(ContractError, match="changed"):
        p.planned_file(root, plan, "shard-0.json")


def test_manual_rejection_blocks_fit_and_does_not_approve(pipeline):
    root, _ = pipeline
    plan = p.load_plan(root, COMMIT)
    for index in range(plan["shards"]):
        p.run_shard(root, root, COMMIT, index, f"100_{index}")
        complete(root, f"100_{index}")
    r.seal_generation(root, root, COMMIT, "100")
    path = root / p.PRIVATE / "review/decisions.json"
    decisions = p.old.read(path)
    decisions["review_complete"] = True
    for row in decisions["decisions"]:
        row.update(dict.fromkeys(MANUAL_REVIEW_CRITERIA, True))
    decisions["decisions"][0][MANUAL_REVIEW_CRITERIA[0]] = False
    atomic_json(path, decisions)
    with pytest.raises(ContractError, match="review failed"):
        r.fit(root, root, COMMIT, "200")
    assert not (root / p.PRIVATE / "fit").exists()


def test_shell_guards_and_container_boundary():
    batch = (ROOT / "scripts/slurm/24h_m2_structured_production.sbatch").read_text()
    launcher = (ROOT / "scripts/slurm/submit_m2_structured_production.sh").read_text()
    assert "CONTAINER_IMAGE=" in batch and "final_container_python" in batch
    assert "final_mark_success" in batch
    assert '--array="0-$((SHARDS-1))%2"' in launcher
    assert "--dependency=\"afterok:$ARRAY\"" in launcher
    assert "--kill-on-invalid-dep=yes" in launcher
    assert "--gres=mps:1" in launcher
    assert '"$MODE" = finalize' in launcher
    assert "submit_m1" not in launcher and "26_m3" not in launcher
