from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.history import validate_requested_splits
from temporal_ehr.final_submission.protocol import (
    begin_test_access,
    freeze_submission_manifest,
    load_frozen_manifest,
    require_submission_hard_gates,
)
from temporal_ehr.final_submission.reporting import _shared_cluster_bootstrap

COMMIT = "a" * 40
TASKS = ("lab_anemia", "guo_readmission", "new_hypertension", "chexpert")


def _frozen_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = Path(__file__).resolve().parents[1]
    copies = (
        "configs/experiments/final_m1_submission.yaml",
        "configs/experiments/final_four_task_m1.yaml",
        "configs/experiments/locked_panel.yaml",
        "reports/generated/final/locked_four_task_m1_v1/final_m1_validation.json",
        "reports/generated/final/locked_four_task_m2_m3_v1/m3_reference.json",
    )
    for relative in copies:
        source = repository / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path, tmp_path / "configs/experiments/final_m1_submission.yaml"


def test_test_split_requires_an_explicit_test_only_authorization() -> None:
    with pytest.raises(ContractError, match="disallowed split"):
        validate_requested_splits(("test",))
    with pytest.raises(ContractError, match="test-only"):
        validate_requested_splits(("train", "test"), allow_test=True)
    assert validate_requested_splits(("test",), allow_test=True) == ("test",)


def test_freeze_precedes_access_and_cannot_be_replaced(tmp_path: Path) -> None:
    root, config = _frozen_fixture(tmp_path)
    gates = require_submission_hard_gates(root, config, COMMIT)
    assert gates["status"] == "PASS"
    assert gates["test_labels_or_performance_read"] is False
    manifest = freeze_submission_manifest(root, config, COMMIT)
    spec, loaded = load_frozen_manifest(root, config, COMMIT)
    assert loaded["frozen_manifest_sha256"] == manifest["frozen_manifest_sha256"]
    started = begin_test_access(spec, loaded)
    assert started["design_changes_after_access_prohibited"] is True
    with pytest.raises(ContractError, match="already started"):
        require_submission_hard_gates(root, config, COMMIT)
    with pytest.raises(ContractError):
        freeze_submission_manifest(root, config, COMMIT)


def _synthetic_predictions() -> list[dict[str, object]]:
    conditions = (
        ("complete_cohort", "C1_correct_compact_time", 0.85),
        ("complete_cohort", "C0_order_only", 0.70),
        ("complete_cohort", "B3_code_count", 0.65),
        ("complete_cohort", "B4_training_prevalence", 0.50),
        ("b1_aligned", "C1_correct_compact_time", 0.85),
        ("b1_aligned", "B1_latest_window_only", 0.62),
        ("b2_aligned", "C1_correct_compact_time", 0.85),
        ("b2_aligned", "B2_latest_code_occurrence", 0.60),
        ("c2_perturbable", "C1_correct_compact_time", 0.85),
        ("c2_perturbable", "C2_reversed_earlier_gaps", 0.66),
        ("c3_three_seed_aggregate", "C1_correct_compact_time", 0.85),
        ("c3_three_seed_aggregate", "C3_three_seed_probability_mean", 0.64),
    )
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(TASKS):
        for subject in range(40):
            target = subject % 2 == 0
            direction = 1.0 if target else -1.0
            for scope, representation, strength in conditions:
                rows.append(
                    {
                        "task": task,
                        "scope": scope,
                        "representation": representation,
                        "perturbation_seed": None,
                        "instance_key": f"{task_index}:{subject}:{scope}:{representation}",
                        "subject_id": subject,
                        "target": target,
                        "probability": 0.5 + direction * strength * (0.2 + subject / 1000),
                    }
                )
    return rows


def test_shared_patient_cluster_bootstrap_and_multiplicity() -> None:
    report = _shared_cluster_bootstrap(
        _synthetic_predictions(),
        tasks=TASKS,
        replicates=40,
        seed=7,
        secondary_names=(
            "C1_minus_C0",
            "C1_minus_C2",
            "C1_minus_B2",
            "C1_minus_B3",
            "C1_minus_B4",
        ),
    )
    assert report["identical_resamples_across_estimands"] is True
    assert report["global_patient_clusters"] == 40
    assert report["co_primary_correction"] == "Holm"
    assert report["per_task_secondary_correction"] == "Benjamini-Hochberg"
    assert all(
        "holm_adjusted_p" in report["equal_weight_panel_macro_contrasts"][name]
        for name in ("C1_minus_B1", "C1_minus_C3_mean")
    )
    assert all(
        "benjamini_hochberg_q" in report["task_contrasts"][task]["C1_minus_C0"]
        for task in TASKS
    )


def test_submission_slurm_graph_is_containerized_and_dependency_locked() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = [
        root / f"scripts/slurm/{name}"
        for name in (
            "29_freeze_m1_submission.sbatch",
            "30_build_m1_test_cohort.sbatch",
            "31_prepare_m1_test_task_array.sbatch",
            "32_embed_m1_test_task_array.sbatch",
            "33_fit_m1_test_report.sbatch",
            "34_build_m1_publication_artifacts.sbatch",
        )
    ]
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "CONTAINER_IMAGE=" in text
        assert "final_container_python" in text
        assert "final_initialize" in text
        assert "final_mark_success" in text
    submit = (root / "scripts/slurm/submit_m1_one_time_test.sh").read_text(encoding="utf-8")
    assert submit.count('dependency="afterok:') == 5
    assert "I_CONFIRM_ONE_TIME_HELD_OUT_EVALUATION" in submit
    assert "git status --porcelain" in submit
    assert "FROZEN_EXPERIMENT_MANIFEST.json" in submit


def test_same_manifest_recovery_never_refreezes_or_refits_completed_results() -> None:
    source = Path("scripts/slurm/resume_m1_one_time_test.sh").read_text(encoding="utf-8")
    assert "I_CONFIRM_RESUME_SAME_FROZEN_HELD_OUT_EVALUATION" in source
    assert "29_freeze_m1_submission.sbatch" not in source
    assert "CURRENT_COMMIT" in source and "MANIFEST_COMMIT" in source
    assert "Refusing recovery while the recorded Prompt 5 graph is still active or pending" in source
    assert 'if [[ -s "${RESULT}" ]]' in source
    result_branch = source.split('if [[ -s "${RESULT}" ]]', maxsplit=1)[1].split("else", maxsplit=1)[0]
    assert "33_fit_m1_test_report.sbatch" not in result_branch


def test_fit_checkpoint_and_publication_directory_are_atomic() -> None:
    reporting = Path("src/temporal_ehr/final_submission/reporting.py").read_text(encoding="utf-8")
    publication = Path("src/temporal_ehr/final_submission/publication.py").read_text(encoding="utf-8")
    assert "AGGREGATE_CHECKPOINT.json" in reporting
    assert "Private held-out fit checkpoint failed identity, hash, or row-count validation" in reporting
    assert reporting.index("atomic_json(checkpoint_path, report)") < reporting.index(
        "_write_immutable_json(destination, report"
    )
    assert 'tempfile.mkdtemp(prefix=".publication."' in publication
    assert "staging_root.replace(publication_root)" in publication


def test_submission_handoff_does_not_allow_private_rows() -> None:
    root = Path(__file__).resolve().parents[1]
    handoff = (root / "scripts/build_handoff_bundle.py").read_text(encoding="utf-8")
    assert "m1_held_out_results.json" in handoff
    assert '"subject_id"' in handoff
    assert '"predictions"' in handoff
    assert "patient_level_artifacts_included" in handoff


def test_frozen_fixture_report_hashes_are_pinned(tmp_path: Path) -> None:
    root, config = _frozen_fixture(tmp_path)
    manifest = freeze_submission_manifest(root, config, COMMIT)
    assert manifest["validation_evidence"]["m1"]["sha256"] == (
        "6d16d6b2494042fd2c84d25ef790f030355738cf8512fb87cf5477f73a3e0c7a"
    )
    assert manifest["validation_evidence"]["m3"]["sha256"] == (
        "7ff518916137317475f8e8a867ea6bcb195cf3c09a27aed67f30b1dc11eb7030"
    )
    payload = json.loads(
        (
            root
            / "reports/generated/submission/locked_four_task_m1_submission_v1/FROZEN_EXPERIMENT_MANIFEST.json"
        ).read_text()
    )
    assert payload["frozen_manifest_sha256"] == manifest["frozen_manifest_sha256"]
