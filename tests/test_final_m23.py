from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import FINAL_TASKS
from temporal_ehr.final_m23.m2 import (
    SummaryCache,
    audit_summary_fidelity,
    build_manual_fidelity_packet,
    m2_embedding_root,
    m2_source_text,
    summary_cache_key,
    validate_manual_fidelity_decisions,
    validate_on_premise_api_base,
)
from temporal_ehr.final_m23.m3 import (
    fixed_projection,
    m3_stage_labels,
    pad_sequences,
    project_then_pad_sequences,
    split_serialized_windows,
    validation_auroc_improved,
)
from temporal_ehr.final_m23.protocol import (
    M3_SEEDS,
    load_final_m23_authorization,
    require_prompt4_hard_gates,
)
from temporal_ehr.final_m23.reporting import _average_seed_predictions, select_approach
from temporal_ehr.serialization.integrity import stable_hash

CONFIG = Path("configs/experiments/final_m2_m3.yaml")


def _spec():
    return load_final_m23_authorization(CONFIG, repository_root=Path.cwd())


def test_prompt4_hard_gate_uses_verified_job451_handoff_and_no_test() -> None:
    result = require_prompt4_hard_gates(Path.cwd(), CONFIG)
    assert result["status"] == "PASS"
    assert result["prompt3_fit_recovery_job_id"] == 451
    assert result["test_predictions_or_performance_used"] is False
    spec = _spec()
    assert spec.tasks == FINAL_TASKS
    assert spec.splits == ("train", "validation")
    assert "test" not in spec.splits


@pytest.mark.parametrize(
    "url",
    [
        "https://10.45.177.17:8000/v1",
        "http://api.openai.com:80/v1",
        "http://127.0.0.1:8000/v1",
        "http://user:secret@10.45.177.17:8000/v1",
    ],
)
def test_m2_rejects_non_authorized_or_credentialed_summary_routes(url: str) -> None:
    with pytest.raises(ContractError):
        validate_on_premise_api_base(url)
    assert validate_on_premise_api_base("http://10.45.177.17:8000/v1") == (
        "http://10.45.177.17:8000/v1/"
    )


def test_summary_cache_key_covers_all_required_generation_inputs() -> None:
    base = {
        "source_hash": "a" * 64,
        "condition": "C1_correct_compact_time",
        "prompt_hash": "b" * 64,
        "model_revision_hash": "c" * 64,
        "decoding_hash": "d" * 64,
    }
    keys = {summary_cache_key(**{**base, field: f"changed-{field}"}) for field in base}
    keys.add(summary_cache_key(**base))
    assert len(keys) == len(base) + 1


def test_m2_embedding_namespace_isolates_source_commits() -> None:
    first = m2_embedding_root(
        _spec(),
        task=FINAL_TASKS[0],
        label="C1_correct_compact_time",
        source_git_commit="a" * 40,
    )
    second = m2_embedding_root(
        _spec(),
        task=FINAL_TASKS[0],
        label="C1_correct_compact_time",
        source_git_commit="b" * 40,
    )
    assert first != second
    assert "source_" + "a" * 40 in first.parts
    with pytest.raises(ContractError):
        m2_embedding_root(
            _spec(),
            task=FINAL_TASKS[0],
            label="C1_correct_compact_time",
            source_git_commit="short",
        )


def test_m2_receives_observation_windows_without_m1_prediction_instruction() -> None:
    instruction = "Represent this record for predicting a future outcome."
    prepared = f"{instruction}\n[TIME=1_30_DAYS_BEFORE_T]\n- code"
    assert m2_source_text(prepared, instruction) == "[TIME=1_30_DAYS_BEFORE_T]\n- code"
    assert m2_source_text(instruction, instruction) == "[NO_PRE_T_EVENTS]"
    with pytest.raises(ContractError, match="locked task instruction"):
        m2_source_text(prepared, "different instruction")


def test_summary_cache_resumes_exact_rows_and_rejects_drift(tmp_path: Path) -> None:
    cache = SummaryCache(tmp_path)
    payload = {
        "cache_key": "a" * 64,
        "source_hash": "b" * 64,
        "condition": "C1_correct_compact_time",
        "prompt_hash": "c" * 64,
        "model_revision_hash": "d" * 64,
        "decoding_hash": "e" * 64,
        "summary": "[TIME=1_30_DAYS_BEFORE_T]\n- Hemoglobin [LOINC/718-7] | value=10 | unit=g/dL",
    }
    payload["summary_hash"] = stable_hash(payload["summary"])
    cache.store(payload["cache_key"], payload)
    expected = {key: value for key, value in payload.items() if key != "summary"}
    assert cache.load(payload["cache_key"], expected) == payload
    with pytest.raises(ContractError, match="immutable inputs"):
        cache.load(payload["cache_key"], {**expected, "prompt_hash": "f" * 64})


def _fidelity_records(*, repair_one: bool = False) -> list[dict[str, object]]:
    records = []
    tasks = FINAL_TASKS
    conditions = (
        "C1_correct_compact_time",
        "C3_shuffled_earlier_time_labels:seed=20260815",
    )
    for task in tasks:
        for condition in conditions:
            for index in range(5):
                source = (
                    "[TIME=31_60_DAYS_BEFORE_T]\n"
                    "- Hemoglobin [LOINC/718-7] | value=10 | unit=g/dL\n"
                    "[TIME=1_30_DAYS_BEFORE_T]\n"
                    "- Hemoglobin [LOINC/718-7] | value=11 | unit=g/dL"
                )
                summary = source + "\nHemoglobin is increasing with recurrence."
                if repair_one and task == tasks[0] and "C3_" in condition and index == 0:
                    summary = summary.replace("[TIME=31_60_DAYS_BEFORE_T]", "[TIME=61_90_DAYS_BEFORE_T]")
                records.append(
                    {
                        "task": task,
                        "condition": condition,
                        "split": "validation",
                        "instance_key": f"{task}-{condition}-{index}",
                        "source_text": source,
                        "summary": summary,
                    }
                )
    return records


def _fidelity_thresholds() -> dict[str, float | int]:
    return {
        "minimum_concept_retention": 0.9,
        "minimum_numeric_retention": 0.95,
        "minimum_unit_retention": 0.95,
        "minimum_recurrence_retention": 0.8,
        "minimum_trend_retention": 0.8,
        "minimum_time_label_retention": 0.95,
        "maximum_unsupported_fact_rate": 0.01,
        "maximum_contradictions": 0,
        "maximum_timing_repairs": 0,
        "maximum_generation_failure_rate": 0.0,
    }


def test_m2_fidelity_is_balanced_and_preserves_falsified_time() -> None:
    report = audit_summary_fidelity(
        _fidelity_records(),
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "PASS"
    assert report["sample_histories"] == 40
    assert report["task_condition_groups"] == 8
    assert report["falsified_timing_repairs"] == 0


def test_m2_fidelity_rejects_repaired_falsified_time() -> None:
    report = audit_summary_fidelity(
        _fidelity_records(repair_one=True),
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "FAIL"
    assert report["falsified_timing_repairs"] == 1


def test_m2_fidelity_reports_structured_fact_contradictions() -> None:
    records = _fidelity_records()
    records[0]["summary"] = str(records[0]["summary"]).replace("value=10", "value=99", 1)
    report = audit_summary_fidelity(
        records,
        thresholds=_fidelity_thresholds(),
        histories_per_task_condition=5,
    )
    assert report["status"] == "FAIL"
    assert report["contradictions"] >= 1


def test_m2_manual_fidelity_requires_all_40_explicit_reviews() -> None:
    packet, decisions = build_manual_fidelity_packet(
        _fidelity_records(),
        histories_per_task_condition=5,
    )
    assert len(packet["entries"]) == 40
    assert all("instance_key" not in row for row in packet["entries"])
    decisions["review_complete"] = True
    for row in decisions["decisions"]:
        for key in tuple(row):
            if key not in {"review_id", "task", "condition"}:
                row[key] = True
    report = validate_manual_fidelity_decisions(packet, decisions)
    assert report["status"] == "PASS"
    assert report["sample_histories"] == 40
    decisions["decisions"][0]["no_timing_repair"] = False
    report = validate_manual_fidelity_decisions(packet, decisions)
    assert report["status"] == "FAIL"
    assert report["criteria_failure_counts"]["no_timing_repair"] == 1


def test_m3_window_split_keeps_64_most_recent_without_reordering() -> None:
    text = "instruction\n" + "\n".join(
        f"[TIME={index * 30 + 1}_{(index + 1) * 30}_DAYS_BEFORE_T]\n- code-{index}"
        for index in range(70, -1, -1)
    )
    blocks = split_serialized_windows(text, maximum_windows=64)
    assert len(blocks) == 64
    assert "code-63" in blocks[0]
    assert "code-0" in blocks[-1]


def test_m3_padding_mask_and_fixed_projection_are_deterministic() -> None:
    first = np.arange(12, dtype=np.float32).reshape(3, 4)
    second = np.arange(8, dtype=np.float32).reshape(2, 4)
    padded, lengths, mask = pad_sequences([first, second], maximum_windows=4)
    assert padded.shape == (2, 4, 4)
    assert lengths.tolist() == [3, 2]
    assert mask.tolist() == [[True, True, True, False], [True, True, False, False]]
    projected_a = fixed_projection(padded, output_dimension=2, seed=20260819)
    projected_b = fixed_projection(padded, output_dimension=2, seed=20260819)
    projected_c = fixed_projection(padded, output_dimension=2, seed=20260820)
    assert np.array_equal(projected_a, projected_b)
    assert not np.array_equal(projected_a, projected_c)
    projected_then_padded, projected_lengths, projected_mask = project_then_pad_sequences(
        [first, second],
        maximum_windows=4,
        output_dimension=2,
        seed=20260819,
    )
    assert np.allclose(projected_then_padded, projected_a)
    assert np.array_equal(projected_lengths, lengths)
    assert np.array_equal(projected_mask, mask)


def test_m3_early_stopping_maximizes_auroc_and_uses_log_loss_only_for_ties() -> None:
    assert validation_auroc_improved(
        current_auroc=0.71,
        best_auroc=0.70,
        current_log_loss=0.70,
        best_log_loss=0.50,
    )
    assert validation_auroc_improved(
        current_auroc=0.70,
        best_auroc=0.70,
        current_log_loss=0.49,
        best_log_loss=0.50,
    )
    assert not validation_auroc_improved(
        current_auroc=0.69,
        best_auroc=0.70,
        current_log_loss=0.40,
        best_log_loss=0.50,
    )


def test_m3_three_seed_patient_alignment_and_probability_average() -> None:
    rows = {
        seed: [
            {
                "task": "lab_anemia",
                "condition": "C1_correct_compact_time",
                "instance_key": f"row-{index}",
                "target": bool(index),
                "probability": 0.1 * index + offset,
            }
            for index in range(2)
        ]
        for seed, offset in zip(M3_SEEDS, (0.0, 0.1, 0.2), strict=True)
    }
    averaged = _average_seed_predictions(rows)
    assert averaged[0]["probability"] == pytest.approx(0.1)
    assert averaged[1]["probability"] == pytest.approx(0.2)
    assert averaged[0]["seeds_averaged"] == list(M3_SEEDS)
    rows[M3_SEEDS[-1]] = rows[M3_SEEDS[-1]][:1]
    with pytest.raises(ContractError, match="identical validation patients"):
        _average_seed_predictions(rows)


def test_m3_sensitivity_is_blocked_before_reference_selection(tmp_path: Path) -> None:
    spec = _spec()
    assert m3_stage_labels(spec, "reference") == ("C1_correct_compact_time",)
    assert spec.m2["sensitivity_conditions"][0] == "B1_latest_window_only"
    assert spec.m3["sensitivity_conditions"][0] == "B1_latest_window_only"
    assert spec.m3["early_stopping_metric"] == "validation_auroc"
    assert spec.m3["embedding_batch_size"] == 32
    assert not (tmp_path / "approach_selection.json").exists()
    # The real spec points only to the Git-ignored governed namespace and currently has no selection.
    if not (spec.aggregate_root / "approach_selection.json").exists():
        with pytest.raises(ContractError, match="completed reference"):
            m3_stage_labels(spec, "sensitivity")


def test_approach_selection_margin_and_no_margin_priority() -> None:
    decisive = {
        "M1": {"macro_auroc": 0.70, "macro_log_loss": 0.60},
        "M2": {"macro_auroc": 0.72, "macro_log_loss": 0.59},
        "M3": {"macro_auroc": 0.71, "macro_log_loss": 0.58},
    }
    assert select_approach(
        decisive,
        minimum_margin=0.005,
        no_margin_priority=("M1", "M2", "M3"),
    )["selected_approach"] == "M2"
    close = {
        **decisive,
        "M2": {"macro_auroc": 0.703, "macro_log_loss": 0.40},
        "M3": {"macro_auroc": 0.702, "macro_log_loss": 0.58},
    }
    assert select_approach(
        close,
        minimum_margin=0.005,
        no_margin_priority=("M1", "M2", "M3"),
    )["selected_approach"] == "M1"


def test_prompt4_array_mapping_is_task_major_and_seed_minor() -> None:
    module_spec = importlib.util.spec_from_file_location("run_final_m23", "scripts/run_final_m23.py")
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    assert module.m3_array_mapping(0, FINAL_TASKS, M3_SEEDS) == ("lab_anemia", 20260815)
    assert module.m3_array_mapping(11, FINAL_TASKS, M3_SEEDS) == ("chexpert", 20260817)
    with pytest.raises(ValueError):
        module.m3_array_mapping(12, FINAL_TASKS, M3_SEEDS)


def test_prompt4_batch_scripts_use_atomic_sentinels_and_no_test_commands() -> None:
    scripts = sorted(Path("scripts/slurm").glob("2[4-8]_*.sbatch"))
    assert [path.name for path in scripts] == [
        "24_m2_summary_task_array.sbatch",
        "25_m2_fit_validation.sbatch",
        "26_m3_task_seed_array.sbatch",
        "27_m3_fit_validation.sbatch",
        "28_freeze_protocol.sbatch",
    ]
    for path in scripts:
        text = path.read_text(encoding="utf-8")
        assert "final_initialize" in text
        assert "final_mark_success" in text
        assert "final_container_python" in text
        assert "test_split" not in text
        assert "OpenAI" not in text
    manual = Path("scripts/slurm/25a_prepare_m2_fidelity_review.sbatch").read_text(
        encoding="utf-8"
    )
    assert "review_m2_fidelity_manual.py prepare" in manual
    assert "final_initialize" in manual
    assert "final_mark_success" in manual
    summary = Path("scripts/slurm/24_m2_summary_task_array.sbatch").read_text(encoding="utf-8")
    embed = Path("scripts/slurm/24b_m2_embed_task_array.sbatch").read_text(encoding="utf-8")
    m3_prepare = Path("scripts/slurm/26a_m3_prepare_task_array.sbatch").read_text(
        encoding="utf-8"
    )
    m3_train = Path("scripts/slurm/26_m3_task_seed_array.sbatch").read_text(encoding="utf-8")
    assert "--summaries-only" in summary
    assert "#SBATCH --gres" not in summary
    assert "#SBATCH --gres=mps:1" in embed
    assert "m3-prepare-task" in m3_prepare
    assert "#SBATCH --array=0-3%1" in m3_prepare
    assert "#SBATCH --array=0-11%2" in m3_train
    assert "TEMPORAL_EHR_MODEL_ROOT" not in m3_train


def test_prompt4_submission_splits_m3_preparation_from_parallel_seed_training() -> None:
    submit = Path("scripts/slurm/submit_prompt4_execution.sh").read_text(encoding="utf-8")
    assert "M3_PREP_ARRAY_JOB_ID" in submit
    assert 'afterok:${M3_PREP_ARRAY_JOB_ID}' in submit
    assert '0-11%${M3_TRAIN_PARALLELISM}' in submit
    assert "PROMPT4_M3_TRAIN_PARALLELISM" in submit
    assert "SENSITIVITY_M3_PREP_ARRAY_JOB_ID" in submit


def test_prompt4_status_helper_is_read_only_and_sentinel_aware() -> None:
    source = Path("scripts/slurm/prompt4_status.sh").read_text(encoding="utf-8")
    assert "squeue" in source
    assert "COMPLETE.json" in source
    assert "FAILED.json" in source
    assert "scancel" not in source
    assert "sbatch" not in source


def test_prompt4_freeze_authorizes_both_co_primary_contrasts() -> None:
    source = Path("src/temporal_ehr/final_m23/reporting.py").read_text(encoding="utf-8")
    assert "C1_correct_compact_time minus B1_latest_window_only" in source
    assert "C1_correct_compact_time minus each of three fixed C3" in source
    submit = Path("scripts/slurm/submit_prompt4_execution.sh").read_text(encoding="utf-8")
    assert "reference-finalize" in submit
    assert "review_m2_fidelity_manual.py finalize" in submit


def test_prompt4_capacity_probe_is_synthetic_only() -> None:
    source = Path("scripts/dgx/benchmark_prompt4_capacity.py").read_text(encoding="utf-8")
    assert "SYNTHETIC_ENGINEERING_CAPACITY_ONLY" in source
    assert "patient_data_used" in source
    assert "EHRSHOT_DATA_PATH" not in source


def test_prompt4_source_contains_no_external_summary_provider_or_patient_artifact() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/temporal_ehr/final_m23").glob("*.py")
    )
    assert "api.openai.com" not in source
    assert "api.anthropic.com" not in source
    assert "outputs/private/final/locked_four_task_m2_m3_v1" in CONFIG.read_text(encoding="utf-8")
    assert "reports/generated/final/locked_four_task_m2_m3_v1" in CONFIG.read_text(encoding="utf-8")
    assert not any(Path("reports/generated/final/locked_four_task_m2_m3_v1").glob("*.json"))


def test_handoff_allowlist_includes_only_prompt4_aggregates_and_config() -> None:
    text = Path("scripts/build_handoff_bundle.py").read_text(encoding="utf-8")
    assert "configs/experiments/final_m2_m3.yaml" in text
    assert "FROZEN_EXPERIMENT_MANIFEST.json" in text
    assert "outputs/private/final/locked_four_task_m2_m3_v1" not in text
