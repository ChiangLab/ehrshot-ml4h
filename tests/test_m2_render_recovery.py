"""Synthetic-only rendering and immutable response-replay regressions."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.final_m23 import render_recovery as recovery
from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_render_v2 as renderer
from temporal_ehr.final_m23 import structured_summary as summary
from temporal_ehr.final_m23.m2 import NonRetryableSummaryError
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = yaml.safe_load((ROOT / old.BASELINE).read_text())["m2"]["fidelity"]


@pytest.mark.parametrize("text", [None, "", "null", "negative", "a\tb\nc", 'x="z"|r=[]',
                                  "[TIME=1_30_DAYS_BEFORE_T]", "μg/mL", "  ", "\\"])
def test_compact_roundtrip_preserves_strings_null_and_order(text):
    source = summary.synthetic_source()
    fact = source["windows"][1]["facts"][0]
    fact["sequence"][0]["text"] = text
    fact["source"] = text
    fact["unit"] = text
    compact = renderer.render(source)
    assert renderer.decode(compact) == source
    assert [w["label"] for w in renderer.decode(compact)["windows"]] == [w["label"] for w in source["windows"]]


def test_empty_and_all_null_runs_roundtrip():
    assert renderer.decode(renderer.render({"windows": []})) == {"windows": []}
    source = summary.ordered_source("I\n[TIME=1_30_DAYS_BEFORE_T]\n- SYN/A\n- SYN/A", "I")
    assert renderer.decode(renderer.render(source)) == source
    assert renderer.decode(renderer.render(source))["windows"][0]["facts"][0]["sequence"][0]["count"] == 2


def test_dense_compaction_does_not_remove_any_facts():
    source = summary.ordered_source("I\n[TIME=1_30_DAYS_BEFORE_T]\n" + "\n".join(
        f"- SYN/{i} | value={i} | source=measurement | unit=mg/dL" for i in range(200)), "I")
    compact = renderer.render(source)
    assert len(compact) < len(summary.render(source))
    assert renderer.decode(compact) == source


@pytest.mark.parametrize("change", ["value", "count", "unit", "time", "order", "omission"])
def test_original_fidelity_gates_still_apply(change):
    source = summary.synthetic_source()
    candidate = copy.deepcopy(source)
    fact = candidate["windows"][0]["facts"][0]
    if change == "value":
        fact["sequence"][0]["value"] = "42"
    elif change == "count":
        fact["sequence"][0]["count"] = 2
    elif change == "unit":
        fact["unit"] = "other"
    elif change == "time":
        candidate["windows"][0]["label"] = "[TIME=61_90_DAYS_BEFORE_T]"
    elif change == "order":
        candidate["windows"].reverse()
    else:
        candidate["windows"][0]["facts"] = []
    with pytest.raises(summary.SummaryContractError):
        renderer.validate(candidate, source, thresholds=THRESHOLDS, token_counter=lambda _: 100)


def test_budget_is_counted_on_new_text_including_legend():
    source = summary.synthetic_source()
    seen = []

    def count(text):
        seen.append(text)
        return 8192

    result = renderer.validate(source, source, thresholds=THRESHOLDS, token_counter=count)
    assert seen == [renderer.render(source)]
    assert result["embedding_tokens"] == 8192
    assert result["roundtrip_exact"] is True
    with pytest.raises(summary.SummaryContractError, match="embedding_token_budget"):
        renderer.validate(source, source, thresholds=THRESHOLDS, token_counter=lambda _: 8193)


@pytest.mark.parametrize("bad", ["bad", renderer.HEADER + "\nSYN/A",
    renderer.HEADER + "\n[TIME=1_30_DAYS_BEFORE_T]\nSYN/A\tn=1\tn=2",
    renderer.HEADER + "\n[TIME=1_30_DAYS_BEFORE_T]\nSYN/A\tr=[]\tv=1"])
def test_malformed_compact_text_rejected(bad):
    with pytest.raises(summary.SummaryContractError):
        renderer.decode(bad)


def test_embedding_identity_versions_rendering_without_changing_generation():
    text = renderer.render(summary.synthetic_source())
    key = renderer.embedding_identity("generation", text, "tokenizer")
    assert key != renderer.embedding_identity("generation", text + "x", "tokenizer")
    assert key != renderer.embedding_identity("generation", text, "other")


class Counter:
    def count(self, text):
        return 200 if text.startswith(renderer.HEADER) else 8265

    def __call__(self, text):
        count = self.count(text)
        if count > 8192:
            raise NonRetryableSummaryError("synthetic token overflow")
        return count


@pytest.fixture
def environment(tmp_path, monkeypatch):
    for path in (old.BASELINE, old.CONFIG, Path("src/temporal_ehr/final_m23/structured_summary.py")):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((ROOT / path).read_text())
    monkeypatch.setattr(old, "old_evidence", lambda _: {"old": "hash"})
    monkeypatch.setattr(old, "tokenizer_fingerprint", lambda _: "tokenizer")
    monkeypatch.setattr(old, "LocalSummaryTokenValidator", lambda *a, **k: Counter())
    config = old.load_config(tmp_path)
    samples = {phase: [{"source": summary.synthetic_source(), "condition": condition, "split": "train",
                       "task": task, "person_hash": f"{phase}-{i}"}
                      for task in old.TASKS for condition in old.CONDITIONS for i in range(4)]
               for phase in ("development", "confirmation")}
    plan = {"config_hash": stable_hash(config), "source_commit": recovery.EXPECTED["source_commit"],
            "served_identity": "served", "tokenizer_hash": "tokenizer", "evidence": {"old": "hash"},
            "upstream_hashes": {}, "prior_canary_people": [], "samples": samples}
    directory = tmp_path / old.PRIVATE / "982"
    old.write_once(directory / "plan.json", plan)
    for i, item in enumerate(samples["development"]):
        item_dir = directory / f"item-{i:02d}"
        old.write_once(item_dir / "source.json", item)
        old.write_once(item_dir / "response-1.json", {"choices": [{"finish_reason": "stop",
                       "message": {"content": summary.compact(item["source"])}}]})
        old.write_once(item_dir / "assessment.json", {"status": "FAIL", "errors": ["embedding_token_budget"]})
    old.write_once(directory / "manifest.json", {
        str(p.relative_to(directory)): file_sha256(p) for p in sorted(directory.rglob("*.json"))})
    expected = {**recovery.EXPECTED, "source_private_manifest_sha256": file_sha256(directory / "manifest.json")}
    monkeypatch.setattr(recovery, "EXPECTED", expected)
    path = tmp_path / recovery.CONFIG
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(expected))
    old.write_once(tmp_path / old.PUBLIC / "development-982.json", {
        "status": "FAIL", "checked": 32, "phase": "development", "job_id": "982",
        "source_commit": expected["source_commit"], "config_hash": stable_hash(config),
        "private_manifest_hash": expected["source_private_manifest_sha256"], "embedding_model_run": False,
        "test_data_used": False, "served_identity": "served", "tokenizer_hash": "tokenizer"})
    old.write_once(tmp_path / "outputs/private/run_status/982/FAILED.json", {
        "job_id": "982", "state": "FAILED", "exit_status": 1, "git_commit": expected["source_commit"]})
    return tmp_path


def complete(root, job):
    old.write_once(root / f"outputs/private/run_status/{job}/COMPLETE.json", {
        "job_id": job, "state": "COMPLETE", "exit_status": 0, "git_commit": "a" * 40})


def test_replay_makes_no_generation_and_preserves_original(environment, monkeypatch):
    root = environment

    def forbidden(*args):
        raise AssertionError("Replay must not contact endpoint")

    monkeypatch.setattr(old, "PrivateClient", forbidden)
    failure = root / "outputs/private/run_status/982/FAILED.json"
    before = file_sha256(failure)
    report = recovery.run(root, root, "984", "replay", "a" * 40)
    assert report["status"] == "PASS"
    assert report["new_generations_requested"] is False
    assert report["feasibility"]["previous_maximum_tokens"] == 8265
    recovery.verify(root, root, "984", "replay", "a" * 40, final=False)
    complete(root, "984")
    assert recovery.verify(root, root, "984", "replay", "a" * 40) == report
    assert file_sha256(failure) == before
    assert "SYN/" not in json.dumps(report)
    with pytest.raises(FileExistsError):
        recovery.run(root, root, "984", "replay", "a" * 40)


def test_confirmation_reuses_frozen_independent_sample(environment, monkeypatch):
    root = environment
    recovery.run(root, root, "984", "replay", "a" * 40)
    complete(root, "984")

    class Client:
        calls = 0

        def __init__(self, config):
            pass

        def identity(self):
            return "served"

        def request(self, path, payload):
            Client.calls += 1
            return {"choices": [{"finish_reason": "stop", "message": {"content": payload["messages"][1]["content"]}}]}

    monkeypatch.setattr(old, "PrivateClient", Client)
    report = recovery.run(root, root, "985", "confirmation", "a" * 40, "984")
    assert report["status"] == "PASS"
    assert Client.calls == 33
    assert report["full_run_authorized"] is False
    complete(root, "985")
    assert recovery.verify(root, root, "985", "confirmation", "a" * 40) == report
    old.write_once(root / "outputs/private/run_status/984/FAILED.json", {"state": "FAILED"})
    with pytest.raises(ContractError, match="conflicting"):
        recovery.verify(root, root, "985", "confirmation", "a" * 40)


@pytest.mark.parametrize("drift", ["response", "config", "generation_module", "confirmation_started", "tokenizer"])
def test_replay_rejects_changed_provenance(environment, monkeypatch, drift):
    root = environment
    if drift == "response":
        (root / old.PRIVATE / "982/item-00/response-1.json").write_text('{}')
    elif drift == "config":
        (root / recovery.CONFIG).write_text('{}')
    elif drift == "generation_module":
        (root / "src/temporal_ehr/final_m23/structured_summary.py").write_text("# drift")
    elif drift == "confirmation_started":
        old.write_once(root / "outputs/private/run_status/983/RUNNING.json", {})
    else:
        monkeypatch.setattr(old, "tokenizer_fingerprint", lambda _: "changed")
    with pytest.raises(ContractError):
        recovery.run(root, root, "984", "replay", "a" * 40)


def test_launch_is_replay_then_confirmation_only():
    launcher = (ROOT / "scripts/slurm/submit_m2_render_recovery.sh").read_text()
    batch = (ROOT / "scripts/slurm/24f_m2_render_recovery.sbatch").read_text()
    assert launcher.count("sbatch --parsable") == 2
    assert '--dependency="afterok:${REPLAY_JOB_ID}"' in launcher
    assert "--kill-on-invalid-dep=yes" in launcher
    assert "--array" not in launcher
    assert "#SBATCH --gres" not in batch
    assert 'final_container_python "$PYTHON" scripts/run_m2_render_recovery.py' in batch
