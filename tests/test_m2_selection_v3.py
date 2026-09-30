"""Synthetic-only tests for the post-993 extractive M2 recovery."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from temporal_ehr.final_m23 import structured_recovery as old
from temporal_ehr.final_m23 import structured_selection_production as production
from temporal_ehr.final_m23 import structured_selection_v3 as selection
from temporal_ehr.final_m23 import structured_v2 as probes
from temporal_ehr.final_m23.structured_summary import SummaryContractError
from temporal_ehr.serialization.integrity import stable_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = old.load_config(ROOT)
THRESHOLDS = yaml.safe_load((ROOT / old.BASELINE).read_text())["m2"]["fidelity"]


def source():
    return copy.deepcopy(probes.probes()[0])


class Client:
    calls = 0

    def __init__(self, response=None):
        self.response = response

    def request(self, route, payload):
        assert route == "chat/completions"
        type(self).calls += 1
        document = self.response
        if document is None:
            indexed = json.loads(payload["messages"][1]["content"])
            document = {
                "windows": [
                    {
                        "window_index": window["window_index"],
                        "fact_indices": [fact["fact_index"] for fact in window["facts"]],
                    }
                    for window in indexed["windows"]
                ]
            }
        return {
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(document)}}
            ]
        }


def test_selection_materializes_exact_source_values_and_order():
    original = source()
    chosen = selection.full_selection(original)
    document = selection.materialize(chosen, original)
    assert document == original
    assert document is not original
    assert document["windows"][0]["facts"][0] is not original["windows"][0]["facts"][0]
    request = selection.payload(original, CONFIG)
    assert request["temperature"] == 0.0
    assert request["response_format"]["type"] == "json_schema"
    assert "clinical codes" in request["messages"][0]["content"]


@pytest.mark.parametrize(
    "mutate,error",
    [
        (
            lambda value: value["windows"][0].__setitem__("window_index", 99),
            "selection_window_order",
        ),
        (
            lambda value: value["windows"][0].__setitem__("fact_indices", [0, 0]),
            "selection_index_order_or_duplicate",
        ),
        (
            lambda value: value["windows"][0].__setitem__("fact_indices", [999]),
            "selection_index_range",
        ),
        (
            lambda value: value["windows"][0].__setitem__("clinical_text", "invented"),
            "selection_window_order",
        ),
    ],
)
def test_malformed_or_content_bearing_selections_fail_closed(mutate, error):
    original = source()
    chosen = selection.full_selection(original)
    mutate(chosen)
    with pytest.raises(SummaryContractError, match=error):
        selection.materialize(chosen, original)
    assert original == source()


def test_assessment_is_bound_to_response_and_never_repairs_source(tmp_path):
    original = source()
    item = {"source": original, "condition": production.C1}
    result = selection.assess(
        item,
        CONFIG,
        THRESHOLDS,
        lambda _: 100,
        tmp_path / "row",
        "served",
        "tokenizer",
        Client(),
    )
    assert result["status"] == "PASS"
    assert result["document"] == original
    assert selection.verify_item(
        tmp_path / "row",
        item,
        CONFIG,
        THRESHOLDS,
        lambda _: 100,
        "served",
        "tokenizer",
    ) == result
    response = tmp_path / "row/response-1.json"
    payload = json.loads(response.read_text())
    payload["choices"][0]["message"]["content"] = '{"windows":[]}'
    response.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        selection.verify_item(
            tmp_path / "row",
            item,
            CONFIG,
            THRESHOLDS,
            lambda _: 100,
            "served",
            "tokenizer",
        )


def test_two_identical_invalid_selections_stop_without_fallback(tmp_path):
    item = {"source": source(), "condition": production.C1}
    start = Client.calls
    result = selection.assess(
        item,
        CONFIG,
        THRESHOLDS,
        lambda _: 100,
        tmp_path / "row",
        "served",
        "tokenizer",
        Client({"windows": []}),
    )
    assert result["status"] == "FAIL"
    assert Client.calls - start == 2
    assert "identical_rejected_selection" in result["errors"]
    assert "summary" not in result


def test_gate_selection_includes_old_failures_and_is_independent():
    rows: dict[str, list[dict]] = {}
    failed: dict[str, dict] = {}
    for task_index, task in enumerate(old.TASKS):
        rows[task] = []
        for index in range(24):
            item = {"source": source(), "condition": production.C1}
            item["source"]["windows"][0]["facts"][0]["code"] = (
                f"SYN/{task_index}/{index}"
            )
            rows[task].append({"split": "train", "condition": production.C1, "item": item})
        rejected = rows[task][0]["item"]
        failed[stable_hash(rejected)] = rejected
    rows["_config"] = CONFIG  # type: ignore[assignment]
    rows["_served"] = "served"  # type: ignore[assignment]
    gates = production.engineering_samples(rows, failed)
    assert len(gates["canary"]) == 32
    assert len(gates["confirmation"]) == 32
    assert not set(gates["canary"]).intersection(gates["confirmation"])
    assert {stable_hash(item) for item in failed.values()} <= {
        stable_hash(item) for item in gates["canary"].values()
    }


def test_shell_graph_is_gated_bounded_and_manual_fit_is_separate():
    launcher = (ROOT / "scripts/slurm/submit_m2_selection_recovery.sh").read_text()
    batch = (ROOT / "scripts/slurm/24j_m2_selection_recovery.sbatch").read_text()
    assert "afterok:$CANARY" in launcher
    assert "afterok:$CONFIRMATION" in launcher
    assert "afterok:$READINESS" in launcher
    assert '0-$((SHARDS - 1))%2' in launcher
    assert launcher.count("--kill-on-invalid-dep=yes") >= 4
    assert '"$MODE" = finalize' in launcher
    assert "M2_SELECTION_PHASE=fit" in launcher
    assert "submit_m1" not in launcher and "26_m3" not in launcher
    assert "final_container_python" in batch
    assert batch.index("scripts/run_m2_selection_recovery.py") < batch.index("final_mark_success")
