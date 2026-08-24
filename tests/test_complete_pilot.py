from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.review_pilot_results import (
    load_pilot_report,
    validate_pilot_report,
    validate_private_postconditions,
)
from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.history import HistoryInstance
from temporal_ehr.experiments import load_pilot_authorization, require_locked_panel
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, file_sha256
from temporal_ehr.pilot.index import finalize_indices, prepare_instance
from temporal_ehr.pilot.model import load_model_identity
from temporal_ehr.pilot.pipeline import _fit_scope
from temporal_ehr.serialization.integrity import stable_hash


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def _event(subject_id: int, time: dt.datetime, code: str) -> dict[str, object]:
    return {
        "subject_id": subject_id,
        "time": time,
        "code": code,
        "numeric_value": None,
        "text_value": None,
        "unit": None,
        "omop_table": "observation",
    }


def _history(*, split: str = "train", future: bool = False) -> HistoryInstance:
    prediction = dt.datetime(2024, 1, 1, 12)
    events = [
        _event(101, prediction - dt.timedelta(days=200), "SYN/OLD"),
        _event(101, prediction - dt.timedelta(days=100), "SYN/MID"),
        _event(101, prediction - dt.timedelta(days=40), "SYN/RECENT"),
        _event(101, prediction - dt.timedelta(days=5), "SYN/LATEST"),
    ]
    events.append(dict(events[-1]))
    if future:
        events.append(_event(101, prediction, "SYN/FUTURE"))
    return HistoryInstance("synthetic-instance", 101, "lab_anemia", split, True, prediction, tuple(events))


def _prepared(history: HistoryInstance) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    prepared = prepare_instance(
        history,
        experiment_id="synthetic_pilot",
        window_days=30,
        c3_seeds=(20260815, 20260816, 20260817),
        descriptions={},
        tokenizer=_WhitespaceTokenizer(),
        instruction="Represent this synthetic record.",
        budget=4096,
        model_hash="m" * 64,
        tokenizer_hash="t" * 64,
    )
    return finalize_indices(
        prepared.index_rows,
        prepared.text_rows,
        experiment_id="synthetic_pilot",
        model_hash="m" * 64,
        tokenizer_hash="t" * 64,
        instructions={"lab_anemia": "Represent this synthetic record."},
        budget=4096,
    )


def test_checked_in_pilot_authorization_is_explicit_and_final_gate_remains_strict() -> None:
    root = Path.cwd()
    spec = load_pilot_authorization(
        root / "configs/experiments/complete_three_task_pilot.yaml",
        repository_root=root,
    )
    assert spec.tasks == ("lab_anemia", "new_acutemi", "guo_los")
    assert spec.splits == ("train", "validation")
    assert spec.token_budget == 4096
    with pytest.raises(ContractError, match="must be reviewed"):
        require_locked_panel(spec.raw)
    assert "outputs/private/pilot" in str(spec.private_root)
    assert str(spec.raw["artifact_namespaces"]["final"]).startswith("outputs/private/final")


def test_pilot_and_final_artifact_namespaces_cannot_overlap(tmp_path: Path) -> None:
    root = Path.cwd()
    for relative in (
        "configs/experiments/pilot_three_task.yaml",
        "configs/experiments/locked_panel.yaml",
        "configs/experiments/complete_three_task_pilot.yaml",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((root / relative).read_text(encoding="utf-8"), encoding="utf-8")
    config_path = tmp_path / "configs/experiments/complete_three_task_pilot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["artifact_namespaces"]["final"] = config["artifact_namespaces"]["pilot"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ContractError, match="completely separate"):
        load_pilot_authorization(config_path, repository_root=tmp_path)


def test_patient_window_index_uniqueness_alignment_content_and_final_labels() -> None:
    indices, texts = _prepared(_history())
    assert len({row["index_row_key"] for row in indices}) == len(indices)
    assert len({row["representation_key"] for row in texts}) == len(texts)
    assert [row["embedding_row_index"] for row in texts] == list(range(len(texts)))
    assert all(row["source_historical_span_days"] > 0 for row in texts)
    assert all(0 <= row["retained_historical_span_days"] <= row["source_historical_span_days"] for row in texts)
    lookup = {row["representation_key"]: row for row in texts}
    for row in indices:
        if row["retention_status"] != "INELIGIBLE":
            assert row["embedding_row_index"] == lookup[row["representation_key"]]["embedding_row_index"]
            assert row["cache_key"] == lookup[row["representation_key"]]["cache_key"]
    full = [
        row
        for row in texts
        if row["representation"]
        in {
            "C0_order_only",
            "C1_correct_compact_time",
            "C2_reversed_earlier_gaps",
            "C3_shuffled_earlier_time_labels",
        }
    ]
    assert len({row["retained_content_hash"] for row in full}) == 1
    assert {row["perturbation_seed"] for row in full if row["representation"].startswith("C3_")} == {
        20260815,
        20260816,
        20260817,
    }
    for representation in (
        "C1_correct_compact_time",
        "C2_reversed_earlier_gaps",
        "C3_shuffled_earlier_time_labels",
    ):
        representation_rows = [row for row in indices if row["representation"] == representation]
        for seed in {row["perturbation_seed"] for row in representation_rows}:
            seeded = [row for row in representation_rows if row["perturbation_seed"] == seed]
            newest = min(seeded, key=lambda row: row["source_window_index"])
            assert newest["displayed_time_label_index"] == newest["correct_time_label_index"]


def test_pilot_rejects_future_events_and_test_split() -> None:
    with pytest.raises(ContractError, match="future or exact"):
        _prepared(_history(future=True))
    with pytest.raises(ContractError, match="test split"):
        _prepared(_history(split="test"))


def test_latest_window_and_latest_code_are_distinct_prepared_baselines() -> None:
    _, texts = _prepared(_history())
    rows = {row["representation"]: row for row in texts}
    latest_window = rows["B1_latest_window_only"]
    latest_code = rows["B2_latest_code_occurrence"]
    assert latest_window["retained_event_count"] == 1
    assert latest_code["retained_event_count"] == 4
    assert latest_window["retained_content_hash"] != latest_code["retained_content_hash"]


def test_resumable_cache_rejects_partial_and_resumes_without_row_drift(tmp_path: Path) -> None:
    keys = ("a", "b", "c", "d")
    cache = ResumableEmbeddingCache(tmp_path / "cache", row_keys=keys, dimension=3, configuration_hash="cfg")
    assert cache.begin_attempt()["attempt_count"] == 1
    assert cache.pending_indices().tolist() == [0, 1, 2, 3]
    cache.write_rows([0, 2], np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
    with pytest.raises(ContractError, match="incomplete"):
        cache.finalize()
    resumed = ResumableEmbeddingCache(tmp_path / "cache", row_keys=keys, dimension=3, configuration_hash="cfg")
    resumed_attempt = resumed.begin_attempt()
    assert resumed_attempt["attempt_count"] == 2
    assert resumed_attempt["failed_attempts"] == resumed_attempt["retries"] == 1
    assert resumed.pending_indices().tolist() == [1, 3]
    resumed.write_rows([1, 3], np.asarray([[0, 0, 1], [1, 1, 1]], dtype=np.float32))
    resumed.finalize()
    assert resumed.finish_attempt()["attempt_active"] is False
    assert resumed.load_complete().shape == (4, 3)
    drifted = ResumableEmbeddingCache(tmp_path / "cache", row_keys=keys, dimension=3, configuration_hash="other")
    with pytest.raises(ContractError, match="does not match"):
        drifted.initialize()
    completed = np.lib.format.open_memmap(cache.completed_path, mode="r+")
    completed[2] = False
    completed.flush()
    with pytest.raises(ContractError, match="bitmap hash drifted"):
        cache.load_complete()


def test_model_identity_requires_pinned_revision_and_exact_artifact_hashes(tmp_path: Path) -> None:
    model = tmp_path / "Qwen3-Embedding-0.6B"
    model.mkdir()
    (model / "PINNED_REVISION.txt").write_text("synthetic-revision\n", encoding="utf-8")
    (model / "tokenizer.json").write_text('{"synthetic": true}\n', encoding="utf-8")
    records = [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(model.iterdir())
    ]
    artifact_hash = stable_hash(
        [(record["path"], record["size_bytes"], record["sha256"]) for record in records]
    )
    (model / "MODEL_ARTIFACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "repository": "Qwen/Qwen3-Embedding-0.6B",
                "revision": "synthetic-revision",
                "artifact_hash": artifact_hash,
                "files": records,
            }
        ),
        encoding="utf-8",
    )
    identity = load_model_identity(model, expected_repository="Qwen/Qwen3-Embedding-0.6B")
    assert identity.artifact_hash == artifact_hash
    (model / "tokenizer.json").write_text("drift\n", encoding="utf-8")
    with pytest.raises(ContractError, match=r"changed size|hash mismatch"):
        load_model_identity(model, expected_repository="Qwen/Qwen3-Embedding-0.6B")


def _fit_rows(*, mismatch: bool = False) -> tuple[list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    vectors: list[list[float]] = []
    for split in ("train", "validation"):
        for index in range(6):
            target = bool(index % 2)
            for representation in ("C0_order_only", "C1_correct_compact_time"):
                row_target = (
                    not target
                    if mismatch and split == "validation" and index == 0 and representation.startswith("C1")
                    else target
                )
                rows.append(
                    {
                        "instance_key": f"{split}-{index}",
                        "task": "synthetic",
                        "split": split,
                        "target": row_target,
                        "representation": representation,
                        "perturbation_seed": None,
                        "embedding_row_index": len(vectors),
                    }
                )
                vectors.append([float(index), float(target), 1.0 if representation.startswith("C1") else 0.0])
    return rows, np.asarray(vectors, dtype=np.float32)


def test_embedding_to_label_joins_are_deterministic_and_fail_on_label_drift() -> None:
    rows, vectors = _fit_rows()
    report, predictions = _fit_scope(
        task="synthetic",
        scope="scope",
        representations=(("C0_order_only", None), ("C1_correct_compact_time", None)),
        rows=rows,
        embeddings=vectors,
        c_grid=(0.1, 1.0),
        random_seed=1,
    )
    assert report["train_instances"] == report["validation_instances"] == 6
    assert len(predictions) == 12
    drifted_rows, drifted_vectors = _fit_rows(mismatch=True)
    with pytest.raises(ContractError, match="label alignment"):
        _fit_scope(
            task="synthetic",
            scope="scope",
            representations=(("C0_order_only", None), ("C1_correct_compact_time", None)),
            rows=drifted_rows,
            embeddings=drifted_vectors,
            c_grid=(0.1,),
            random_seed=1,
        )


def _metrics() -> dict[str, float]:
    return {"auroc": 0.5, "auprc": 0.5, "brier_score": 0.25, "log_loss": 0.693}


def _fit_report() -> dict[str, object]:
    return {"metrics": _metrics(), "calibration": {"status": "SUPPRESSED"}}


def _pilot_aggregate_fixture() -> dict[str, object]:
    seeds = (20260815, 20260816, 20260817)
    tasks = ("lab_anemia", "new_acutemi", "guo_los")
    scopes = {
        "complete_history": {
            "B1_latest_window_only",
            "B2_latest_code_occurrence",
            "C0_order_only",
            "C1_correct_compact_time",
        },
        "c2_perturbable": {
            "C0_order_only",
            "C1_correct_compact_time",
            "C2_reversed_earlier_gaps",
        },
        **{
            f"c3_seed_{seed}": {
                "C0_order_only",
                "C1_correct_compact_time",
                f"C3_shuffled_earlier_time_labels:seed={seed}",
            }
            for seed in seeds
        },
    }
    task_report = {
        "scopes": {
            name: {
                "train_instances": 4,
                "validation_instances": 4,
                "representations": {representation: _fit_report() for representation in representations},
            }
            for name, representations in scopes.items()
        },
        "B4_training_prevalence": {
            "metrics": _metrics(),
            "calibration": {"status": "SUPPRESSED"},
        },
        "B3_governed_code_count": {
            "metrics": {"model": _metrics()},
            "calibration": {"status": "SUPPRESSED"},
        },
        "paired_contrasts": [
            {
                "first": "C1_correct_compact_time",
                "second": second,
                "estimate": 0.0,
                "ci_95": [-0.1, 0.1],
            }
            for second in (
                "B4_training_prevalence",
                "B3_governed_code_count",
                "B1_latest_window_only",
                "B2_latest_code_occurrence",
                "C0_order_only",
                "C2_reversed_earlier_gaps",
            )
        ],
        "c3_seed_contrasts": [
            {
                "first": "C1_correct_compact_time",
                "second": f"C3_shuffled_earlier_time_labels:seed={seed}",
                "estimate": 0.0,
                "ci_95": [-0.1, 0.1],
            }
            for seed in seeds
        ],
        "c3_seed_summary": {"seeds": list(seeds)},
    }
    return {
        "status": "PASS",
        "protocol_decision": "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED",
        "experiment_config_hash": "c" * 64,
        "source_git_commit": "a" * 40,
        "scientific_scope": {
            "tasks": list(tasks),
            "splits": ["train", "validation"],
            "model": "Qwen/Qwen3-Embedding-0.6B",
            "model_revision_hash": "d" * 64,
            "model_artifact_hash": "e" * 64,
            "tokenizer_hash": "f" * 64,
            "token_budget": 4096,
            "approach": "M1 direct serialization only",
            "baselines": ["B4", "B3", "B1", "B2"],
            "conditions": ["C0", "C1", "C2", "C3"],
            "c3_seeds": list(seeds),
        },
        "content_matching_passed": True,
        "final_panel_used_for_pilot": False,
        "test_predictions_or_performance_used": False,
        "m2_run": False,
        "m3_run": False,
        "qwen_8b_run": False,
        "private_predictions_sha256": "b" * 64,
        "private_prediction_rows": 42,
        "prepare": {
            "status": "PASS",
            "test_predictions_or_performance_used": False,
            "task_summaries": [
                {
                    "task": task,
                    "content_matching_passed": True,
                    "test_split_used": False,
                    "representation_coverage": {
                        "C1_correct_compact_time": {
                            "eligible_histories": 4,
                            "retained_windows": 8,
                            "retained_events": 16,
                            "token_count": {"median": 20.0},
                        }
                    },
                }
                for task in tasks
            ],
        },
        "throughput": {
            "status": "PASS",
            "failures": 0,
            "retries": 0,
            "test_outcomes_used": False,
        },
        "embedding_tasks": [
            {"task": task, "status": "PASS", "cache_complete": True, "failures": 0, "retries": 0}
            for task in tasks
        ],
        "tasks": {task: task_report for task in tasks},
    }


def test_pilot_reviewer_accepts_only_complete_privacy_safe_aggregate_configured_path(tmp_path: Path) -> None:
    report = tmp_path / "reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps(_pilot_aggregate_fixture()) + "\n", encoding="utf-8")
    assert load_pilot_report(report, tmp_path)["status"] == "PASS"
    with pytest.raises(ContractError, match="reports/generated"):
        load_pilot_report(tmp_path / "outputs/private/pilot/results.json", tmp_path)
    unsafe = _pilot_aggregate_fixture()
    unsafe["subject_id"] = 123
    with pytest.raises(ContractError, match="row-level"):
        validate_pilot_report(unsafe)


def test_pilot_private_prediction_postcondition_is_hash_bound(tmp_path: Path) -> None:
    predictions = (
        tmp_path
        / "outputs/private/pilot/three_task_m1_engineering_pilot_v1/fit/validation_predictions.parquet"
    )
    predictions.parent.mkdir(parents=True)
    predictions.write_bytes(b"synthetic-private-predictions")
    payload = _pilot_aggregate_fixture()
    payload["private_predictions_sha256"] = file_sha256(predictions)
    validate_private_postconditions(payload, tmp_path)
    predictions.write_bytes(b"drift")
    with pytest.raises(ContractError, match="hash-mismatched"):
        validate_private_postconditions(payload, tmp_path)


def test_slurm_pilot_chain_is_containerized_serial_and_fail_closed() -> None:
    scripts = {
        name: (Path("scripts/slurm") / name).read_text(encoding="utf-8")
        for name in (
            "12_pilot_prepare.sbatch",
            "13_pilot_throughput.sbatch",
            "14_pilot_embed_task_array.sbatch",
            "15_pilot_fit_report.sbatch",
        )
    }
    for text in scripts.values():
        assert "#SBATCH --partition=batch" in text
        assert "pilot_container_python" in text
        assert "pilot_initialize" in text and "pilot_mark_success" in text
    assert "#SBATCH --gres=mps:1" not in scripts["12_pilot_prepare.sbatch"]
    assert "#SBATCH --gres=mps:1" in scripts["13_pilot_throughput.sbatch"]
    assert "#SBATCH --gres=mps:1" in scripts["14_pilot_embed_task_array.sbatch"]
    assert "#SBATCH --array=0-2%1" in scripts["14_pilot_embed_task_array.sbatch"]
    assert "SLURM_ARRAY_JOB_ID" in scripts["14_pilot_embed_task_array.sbatch"]
    assert scripts["12_pilot_prepare.sbatch"].index("pilot_initialize") < scripts[
        "12_pilot_prepare.sbatch"
    ].index("for path in")
    common = Path("scripts/slurm/pilot_job_common.sh").read_text(encoding="utf-8")
    assert "RUNNING" in common and "COMPLETE" in common and "FAILED" in common
    assert "export PILOT_GIT_COMMIT" in common
    submit = Path("scripts/slurm/submit_pilot_chain.sh").read_text(encoding="utf-8")
    assert submit.count("--dependency=\"afterok:") == 4
    assert "THROUGHPUT_JOB_ID" in submit and "EMBED_ARRAY_JOB_ID" in submit and "REPORT_JOB_ID" in submit
    assert "PILOT_AFTEROK_JOB_ID" in submit

    prompt2_submit = Path("scripts/slurm/submit_prompt2_execution.sh").read_text(encoding="utf-8")
    assert "git status --porcelain" in prompt2_submit
    assert 'PILOT_AFTEROK_JOB_ID="${SMOKE_JOB_ID}"' in prompt2_submit
    assert "MODEL_ARTIFACT_MANIFEST.json" in prompt2_submit
    assert "outputs/private/submissions/prompt2-${SMOKE_JOB_ID}.env" in prompt2_submit
    assert "Refusing to overwrite existing submission record" in prompt2_submit
