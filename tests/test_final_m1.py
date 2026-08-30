from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from scripts.build_handoff_bundle import _aggregate_summary, assert_handoff_payload_safe
from scripts.review_final_m1_results import validate_final_m1_report
from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.history import HistoryInstance
from temporal_ehr.experiments import FINAL_TASKS, load_final_m1_authorization
from temporal_ehr.final_m1.cohort import build_final_cohort, selection_rank
from temporal_ehr.final_m1.embedding import load_final_condition_embeddings
from temporal_ehr.final_m1.foundation import _metadata_rows, validate_outcome_contracts
from temporal_ehr.final_m1.gates import require_prompt3_hard_gates
from temporal_ehr.final_m1.report import fit_condition_scope
from temporal_ehr.final_m1.throughput import plan_confirmatory_cells
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, file_sha256
from temporal_ehr.pilot.index import finalize_indices, prepare_instance
from temporal_ehr.pilot.model import ModelIdentity


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def _parquet_bytes(table: pa.Table) -> bytes:
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def _write_synthetic_assets(root: Path, *, invert_outcomes: bool = False) -> None:
    root.mkdir(parents=True)
    times = [dt.datetime(2020, 1, 1), dt.datetime(2021, 1, 1), dt.datetime(2022, 1, 1)]
    split_table = pa.table(
        {
            "subject_id": pa.array([101, 202, 303], type=pa.int64()),
            "split": pa.array(["train", "tuning", "held_out"]),
        }
    )
    fields = {
        "lab_anemia": ("integer_value", pa.int64()),
        "guo_readmission": ("boolean_value", pa.bool_()),
        "new_hypertension": ("boolean_value", pa.bool_()),
        "chexpert": ("integer_value", pa.int64()),
    }
    with zipfile.ZipFile(root / "synthetic-ehrshot.zip", "w") as bundle:
        bundle.writestr("metadata/subject_splits.parquet", _parquet_bytes(split_table))
        for task, (field, arrow_type) in fields.items():
            subjects = [101, 101, 101, 202, 202, 303]
            prediction_times = [*times, times[0], times[1], times[0]]
            if field == "boolean_value":
                values = [False, True, False, True, False, True]
                if invert_outcomes:
                    values = [not value for value in values]
            else:
                values = [0, 1, 2, 0, 1, 2]
                if task == "chexpert":
                    values = [8192, 1, 2, 8192, 1, 2]
                if invert_outcomes:
                    values = list(reversed(values))
            table = pa.table(
                {
                    "subject_id": pa.array(subjects, type=pa.int64()),
                    "prediction_time": pa.array(prediction_times, type=pa.timestamp("us")),
                    field: pa.array(values, type=arrow_type),
                }
            )
            bundle.writestr(f"labels/{task}/labels.parquet", _parquet_bytes(table))


def _spec_at(tmp_path: Path):
    spec = load_final_m1_authorization(
        Path("configs/experiments/final_four_task_m1.yaml"),
        repository_root=Path.cwd(),
    )
    private = tmp_path / "outputs/private/final/locked_four_task_m1_v1"
    return replace(
        spec,
        cohort_path=private / "cohort/final_panel_cohort_v1.parquet",
        private_root=private,
        aggregate_root=tmp_path / "reports/generated/final/locked_four_task_m1_v1",
    )


def test_final_authorization_is_exact_and_outcome_contracts_match_registry() -> None:
    spec = load_final_m1_authorization(
        Path("configs/experiments/final_four_task_m1.yaml"), repository_root=Path.cwd()
    )
    assert spec.tasks == FINAL_TASKS
    assert spec.splits == ("train", "validation")
    assert spec.token_budget == 8192
    assert "outputs/private/final/locked_four_task_m1_v1" in str(spec.private_root)
    assert "outputs/private/pilot" not in str(spec.private_root)
    registry = yaml.safe_load(Path("configs/tasks.yaml").read_text(encoding="utf-8"))
    audits = validate_outcome_contracts(spec, registry)
    assert set(audits) == set(FINAL_TASKS)
    assert all(item["target_defining_future_events_retained"] == 0 for item in audits.values())


def test_selection_rank_includes_every_prespecified_component() -> None:
    values = {
        "dataset_release_id": "release-a",
        "task": "lab_anemia",
        "split": "train",
        "subject_id": 101,
        "frozen_seed": 20260815,
        "sorted_prediction_time_index": 0,
    }
    baseline = selection_rank(**values)
    assert len(baseline) == 64
    alternatives = (
        {**values, "dataset_release_id": "release-b"},
        {**values, "task": "chexpert"},
        {**values, "split": "validation"},
        {**values, "subject_id": 202},
        {**values, "frozen_seed": 20260816},
        {**values, "sorted_prediction_time_index": 1},
    )
    assert all(selection_rank(**alternative) != baseline for alternative in alternatives)


def test_final_cohort_is_one_per_patient_task_outcome_independent_and_excludes_test(tmp_path: Path) -> None:
    registry = yaml.safe_load(Path("configs/tasks.yaml").read_text(encoding="utf-8"))
    assets_a = tmp_path / "assets-a"
    assets_b = tmp_path / "assets-b"
    _write_synthetic_assets(assets_a)
    _write_synthetic_assets(assets_b, invert_outcomes=True)
    spec_a = _spec_at(tmp_path / "run-a")
    spec_b = _spec_at(tmp_path / "run-b")
    report_a = build_final_cohort(asset_root=assets_a, registry=registry, spec=spec_a)
    report_b = build_final_cohort(asset_root=assets_b, registry=registry, spec=spec_b)
    rows_a = pq.read_table(spec_a.cohort_path).to_pylist()
    rows_b = pq.read_table(spec_b.cohort_path).to_pylist()
    assert len(rows_a) == len({(row["subject_id"], row["task"]) for row in rows_a})
    assert {row["split"] for row in rows_a} == {"train", "validation"}
    assert 303 not in {row["subject_id"] for row in rows_a}
    ranks_a = {(row["subject_id"], row["task"]): row["selection_hash"] for row in rows_a}
    ranks_b = {(row["subject_id"], row["task"]): row["selection_hash"] for row in rows_b}
    assert ranks_a == ranks_b
    assert report_a["test_instances_selected"] == report_b["test_instances_selected"] == 0
    with pytest.raises(ContractError, match=r"hash-mismatched|stale"):
        spec_a.cohort_path.write_bytes(b"drift")
        build_final_cohort(asset_root=assets_a, registry=registry, spec=spec_a)


def test_empty_history_still_has_complete_c0_c1_embedding_metadata() -> None:
    prediction = dt.datetime(2024, 1, 1)
    instance = HistoryInstance(
        "synthetic-empty",
        101,
        "lab_anemia",
        "train",
        False,
        prediction,
        (),
    )
    prepared = prepare_instance(
        instance,
        experiment_id="locked_four_task_m1_v1",
        window_days=30,
        c3_seeds=(20260815, 20260816, 20260817),
        descriptions={},
        tokenizer=_WhitespaceTokenizer(),
        instruction="Represent this empty synthetic record.",
        budget=8192,
        model_hash="m" * 64,
        tokenizer_hash="t" * 64,
    )
    indices, texts = finalize_indices(
        prepared.index_rows,
        prepared.text_rows,
        experiment_id="locked_four_task_m1_v1",
        model_hash="m" * 64,
        tokenizer_hash="t" * 64,
        instructions={"lab_anemia": "Represent this empty synthetic record."},
        budget=8192,
    )
    assert indices == []
    assert {row["representation"] for row in texts} == {"C0_order_only", "C1_correct_compact_time"}
    assert all(row["token_count"] == 5 for row in texts)
    metadata = _metadata_rows(texts)
    assert len(metadata) == len(texts)
    assert all("text" not in row and row["mirrors_embedding_row"] for row in metadata)


def test_runtime_only_confirmatory_plan_is_prefix_ordered_with_two_x_contingency() -> None:
    counts = {
        task: {
            "__primary_total__": 100,
            "C1_correct_compact_time": 10,
            "C3_shuffled_earlier_time_labels": 20,
            "B1_latest_window_only": 10,
            "B2_latest_code_occurrence": 10,
        }
        for task in FINAL_TASKS
    }
    priorities = (
        "C1_correct_compact_time",
        "C3_shuffled_earlier_time_labels",
        "B1_latest_window_only",
        "B2_latest_code_occurrence",
    )
    plan = plan_confirmatory_cells(
        row_counts_by_task=counts,
        primary_documents_per_second=10.0,
        confirmatory_documents_per_second={priority: 1.0 for priority in priorities},
        priorities=priorities,
        contingency_multiplier=2.0,
        task_job_time_limit_seconds=100,
        minimum_runtime_reserve_seconds=10,
    )
    assert plan["primary_feasible_with_2x_contingency"]
    assert plan["contingency_multiplier"] == 2.0
    assert plan["outcome_performance_used_for_feasibility"] is False
    assert all(
        decision["confirmatory_priorities_selected"] == list(priorities[:2])
        for decision in plan["tasks"].values()
    )


def _condition_data(*, mismatched: bool = False):
    metadata = []
    vectors = []
    for split in ("train", "validation"):
        for index in range(8):
            target = bool(index % 2)
            metadata.append(
                {
                    "instance_key": f"{split}-{index}",
                    "split": split,
                    "target": not target if mismatched and split == "validation" and index == 0 else target,
                    "condition_embedding_row_index": len(vectors),
                }
            )
            vectors.append([float(index), float(target), 1.0])
    return metadata, np.asarray(vectors, dtype=np.float32)


def test_final_scope_fit_is_patient_aligned_and_rejects_label_drift() -> None:
    c0 = _condition_data()
    c1 = _condition_data()
    report, predictions = fit_condition_scope(
        task="synthetic",
        scope="complete_cohort",
        condition_data={"C0_order_only": c0, "C1_correct_compact_time": c1},
        c_grid=(0.1, 1.0),
        random_seed=1,
        model_role="primary",
    )
    assert report["train_instances"] == report["validation_instances"] == 8
    assert len(predictions) == 16
    with pytest.raises(ContractError, match="alignment"):
        fit_condition_scope(
            task="synthetic",
            scope="complete_cohort",
            condition_data={"C0_order_only": c0, "C1_correct_compact_time": _condition_data(mismatched=True)},
            c_grid=(0.1,),
            random_seed=1,
            model_role="primary",
        )


def test_final_cache_loader_rejects_model_identity_drift(tmp_path: Path) -> None:
    spec = _spec_at(tmp_path)
    identity = ModelIdentity(
        repository="Qwen/Qwen3-Embedding-0.6B",
        revision="revision",
        artifact_hash="a" * 64,
        tokenizer_hash="t" * 64,
        directory=tmp_path / "model",
    )
    root = spec.private_root / "embeddings/primary_0.6b/lab_anemia/C1_correct_compact_time"
    root.mkdir(parents=True)
    row_key = "r" * 64
    configuration_hash = "c" * 64
    metadata = [
        {
            "experiment_id": spec.experiment_id,
            "task": "lab_anemia",
            "split": "validation",
            "instance_key": "synthetic",
            "subject_id": 101,
            "prediction_time": dt.datetime(2024, 1, 1),
            "target": False,
            "representation": "C1_correct_compact_time",
            "perturbation_seed": None,
            "condition_embedding_row_index": 0,
            "condition_cache_key": row_key,
            "text_hash": "h" * 64,
            "model_role": "primary",
            "model_repository": identity.repository,
            "model_artifact_hash": identity.artifact_hash,
            "tokenizer_hash": identity.tokenizer_hash,
            "mirrors_embedding_row": True,
        }
    ]
    metadata_path = root / "patient_window_metadata.parquet"
    pq.write_table(pa.Table.from_pylist(metadata), metadata_path)
    cache = ResumableEmbeddingCache(
        root,
        row_keys=[row_key],
        dimension=2,
        configuration_hash=configuration_hash,
    )
    cache.write_rows([0], np.asarray([[1.0, 0.0]], dtype=np.float32))
    cache.finalize()
    (root / "SUMMARY.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": spec.experiment_id,
                "task": "lab_anemia",
                "condition": "C1_correct_compact_time",
                "model_role": "primary",
                "model_repository": identity.repository,
                "model_artifact_hash": identity.artifact_hash,
                "tokenizer_hash": identity.tokenizer_hash,
                "configuration_hash": configuration_hash,
                "rows": 1,
                "dimension": 2,
                "metadata_sha256": file_sha256(metadata_path),
                "cache_complete_sha256": file_sha256(cache.complete_path),
            }
        ),
        encoding="utf-8",
    )
    rows, vectors = load_final_condition_embeddings(
        spec=spec,
        task="lab_anemia",
        label="C1_correct_compact_time",
        role="primary",
        identity=identity,
    )
    assert len(rows) == 1 and vectors.shape == (1, 2)
    drifted = replace(identity, artifact_hash="b" * 64)
    with pytest.raises(ContractError, match="identity"):
        load_final_condition_embeddings(
            spec=spec,
            task="lab_anemia",
            label="C1_correct_compact_time",
            role="primary",
            identity=drifted,
        )


def test_prompt3_hard_gates_require_recovery_pilot_hash_and_no_test_use(tmp_path: Path) -> None:
    for relative in ("configs/experiments/locked_panel.yaml", "configs/experiments/final_four_task_m1.yaml"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(Path(relative).read_text(encoding="utf-8"), encoding="utf-8")
    pilot_report = tmp_path / "reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json"
    pilot_report.parent.mkdir(parents=True)
    pilot_report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_decision": "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED",
                "test_predictions_or_performance_used": False,
                "final_panel_used_for_pilot": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(pilot_report.read_bytes()).hexdigest()
    config_path = tmp_path / "configs/experiments/final_four_task_m1.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["preconditions"]["pilot_report_sha256"] = digest
    config["preconditions"]["pilot_source_commit"] = "a" * 40
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    handoff_root = tmp_path / "reports/generated/handoff"
    handoff_root.mkdir(parents=True)
    (handoff_root / "project_handoff.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "408",
                        "sentinels": [
                            {
                                "sentinel": "COMPLETE.json",
                                "state": "COMPLETE",
                                "exit_status": 0,
                                "sha256": "b" * 64,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pilot_handoff = {
        "git": {"head_commit": "a" * 40},
        "aggregate_reports": [
            {
                "path": "reports/generated/pilot/three_task_m1_engineering_pilot_v1/pilot_results.json",
                "status": "PASS",
                "protocol_decision": "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED",
                "sha256": digest,
                "test_predictions_or_performance_used": False,
                "final_panel_used_for_pilot": False,
            }
        ],
    }
    (handoff_root / "pilot_handoff.json").write_text(json.dumps(pilot_handoff), encoding="utf-8")
    assert require_prompt3_hard_gates(tmp_path, config_path)["status"] == "PASS"
    pilot_handoff["aggregate_reports"][0]["test_predictions_or_performance_used"] = True
    (handoff_root / "pilot_handoff.json").write_text(json.dumps(pilot_handoff), encoding="utf-8")
    with pytest.raises(ContractError, match=r"test|hard-gate"):
        require_prompt3_hard_gates(tmp_path, config_path)


def test_final_slurm_chain_is_containerized_task_mapped_and_atomic() -> None:
    names = (
        "20_final_foundation.sbatch",
        "21_final_throughput.sbatch",
        "22_final_m1_embed_task_array.sbatch",
        "23_final_m1_fit_validation.sbatch",
    )
    scripts = {name: (Path("scripts/slurm") / name).read_text(encoding="utf-8") for name in names}
    for text in scripts.values():
        assert "#SBATCH --partition=batch" in text
        assert "#SBATCH --nodelist=dgit00038673" in text
        assert "final_initialize" in text and "final_container_python" in text and "final_mark_success" in text
    assert "#SBATCH --array=0-3%1" in scripts["22_final_m1_embed_task_array.sbatch"]
    assert "SLURM_ARRAY_JOB_ID" in scripts["22_final_m1_embed_task_array.sbatch"]
    assert "--time=12:00:00" in scripts["20_final_foundation.sbatch"]
    assert "--time=12:00:00" in scripts["22_final_m1_embed_task_array.sbatch"]
    assert "--time=12:00:00" in scripts["23_final_m1_fit_validation.sbatch"]
    common = Path("scripts/slurm/final_job_common.sh").read_text(encoding="utf-8")
    assert "RUNNING" in common and "COMPLETE" in common and "FAILED" in common
    assert "export FINAL_M1_GIT_COMMIT" in common
    submit = Path("scripts/slurm/submit_prompt3_execution.sh").read_text(encoding="utf-8")
    assert "git status --porcelain" in submit
    assert "Qwen3-Embedding-8B" in submit
    assert "--job-name=tehr-prompt3-preflight" in submit
    assert '--nodelist="${EXECUTION_NODE}"' in submit
    assert '--container-image="${CONTAINER_IMAGE}"' in submit
    assert "env PYTHONPATH=src .venv/bin/python scripts/run_final_m1.py hard-gates" in submit
    assert 'afterok:${FOUNDATION_JOB_ID}:${SMOKE_8B_JOB_ID}' in submit
    assert 'afterok:${THROUGHPUT_JOB_ID}' in submit
    assert 'afterok:${EMBED_ARRAY_JOB_ID}' in submit
    assert "Refusing to overwrite existing submission record" in submit
    smoke_8b = Path("scripts/slurm/06_smoke_qwen_8b.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --nodelist=dgit00038673" in smoke_8b


def test_final_report_reviewer_fails_closed_on_test_or_row_content() -> None:
    with pytest.raises(ContractError, match=r"test|scope"):
        validate_final_m1_report(
            {
                "status": "PASS",
                "protocol_decision": "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED",
                "content_matching_passed": True,
                "label_index_only_perturbations_passed": True,
                "patient_window_metadata_alignment_passed": True,
                "pilot_artifacts_reused": False,
                "test_predictions_or_performance_used": True,
                "test_results_reported": False,
                "m2_run": False,
                "m3_run": False,
            }
        )
    with pytest.raises(ContractError, match="row field"):
        validate_final_m1_report({"subject_id": 12345})


def test_final_handoff_summary_excludes_metrics_and_private_rows(tmp_path: Path) -> None:
    report = tmp_path / "final_m1_validation.json"
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "experiment_id": "locked_four_task_m1_v1",
                "protocol_decision": "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED",
                "scientific_scope": {"tasks": list(FINAL_TASKS)},
                "content_matching_passed": True,
                "test_predictions_or_performance_used": False,
                "test_results_reported": False,
                "pilot_artifacts_reused": False,
                "metrics": {"auroc": 0.8123456789},
                "private_rows": [
                    {
                        "subject_id": 12345,
                        "prediction_time": "2026-08-29T00:00:00Z",
                        "prediction": 0.91,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = _aggregate_summary(report)
    serialized = json.dumps(summary, sort_keys=True)
    assert summary["tasks"] == list(FINAL_TASKS)
    assert "metrics" not in summary and "private_rows" not in summary
    assert "0.8123456789" not in serialized and "12345" not in serialized
    assert_handoff_payload_safe({"aggregate_reports": [summary]})
