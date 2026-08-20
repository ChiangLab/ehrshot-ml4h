from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.review_phase1_results import load_aggregate_report
from temporal_ehr.data.contracts import ContractError
from temporal_ehr.data.task_registry import load_task_registry
from temporal_ehr.experiments import load_experiment_panel, require_locked_panel, window_days_for_task
from temporal_ehr.run_status import write_final, write_running
from temporal_ehr.serialization.blocks import ClinicalEvent, exact_duplicate_key
from temporal_ehr.serialization.integrity import (
    build_instance_manifest,
    require_aggregate_output,
    require_private_output,
)
from temporal_ehr.serialization.qc import run_protocol_alignment_qc


def test_checked_in_panel_statuses_and_fail_closed_gate() -> None:
    expected = {
        "pilot_three_task.yaml": "PILOT",
        "proposed_four_task.yaml": "PROPOSED",
        "locked_panel.yaml": "PENDING",
    }
    for name, status in expected.items():
        panel = load_experiment_panel(Path("configs/experiments") / name)
        assert panel["status"] == status
        with pytest.raises(ContractError, match="embedding training is blocked"):
            require_locked_panel(panel)


def test_locked_panel_gate_accepts_only_nonempty_reviewed_panel(tmp_path: Path) -> None:
    path = tmp_path / "locked.yaml"
    path.write_text("status: LOCKED\ntasks: [lab_anemia, chexpert]\n", encoding="utf-8")
    assert require_locked_panel(load_experiment_panel(path)) == ("lab_anemia", "chexpert")
    path.write_text("status: LOCKED\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ContractError, match="at least one task"):
        load_experiment_panel(path)


def test_window_size_is_selected_from_configured_family_not_task_name() -> None:
    registry = load_task_registry(Path("configs/tasks.yaml"))
    assert window_days_for_task("lab_anemia", registry) == 30
    assert window_days_for_task("guo_los", registry) == 30
    assert window_days_for_task("chexpert", registry) == 30
    assert window_days_for_task("new_hypertension", registry) == 90
    with pytest.raises(ContractError, match="No configured serialization window"):
        window_days_for_task("invented_task", registry)


def test_private_and_aggregate_paths_fail_closed(tmp_path: Path) -> None:
    private = tmp_path / "outputs/private/manifests/records.jsonl"
    aggregate = tmp_path / "reports/generated/review/report.json"
    assert require_private_output(private, tmp_path) == private
    assert require_aggregate_output(aggregate, tmp_path) == aggregate
    with pytest.raises(ContractError, match="outputs/private"):
        require_private_output(aggregate, tmp_path)
    with pytest.raises(ContractError, match="reports/generated"):
        require_aggregate_output(private, tmp_path)


def test_integrity_manifest_is_hash_only_for_private_identity() -> None:
    prediction = dt.datetime(2024, 1, 1)
    event = ClinicalEvent(prediction - dt.timedelta(days=1), "SYN/A", subject_id=123456)
    manifest = build_instance_manifest(
        dataset_revision="dataset-revision",
        config_revision="config-revision",
        model_revision="no-model",
        tokenizer_revision="tokenizer-revision",
        task="synthetic_task",
        split="train",
        outcome=True,
        subject_id=123456,
        instance_key="synthetic-instance",
        prediction_time=prediction,
        source_events=(event,),
        deduplicated_events=(event,),
        retained_signature=(exact_duplicate_key(event),),
        retained_block_signature=((0, (exact_duplicate_key(event),)),),
        rendered_text="[TIME=1_30_DAYS_BEFORE_T]\n- SYN/A",
        condition="C1_correct_compact_time",
        perturbation_seed=None,
        comparison_seed=20260815,
        token_budget=4096,
    )
    text = json.dumps(manifest)
    assert "123456" not in text
    assert prediction.isoformat() not in text
    assert "synthetic-instance" not in text
    for field in (
        "dataset_revision_hash",
        "config_revision_hash",
        "model_revision_hash",
        "tokenizer_revision_hash",
        "patient_hash",
        "prediction_instance_hash",
        "source_events_hash",
        "deduplicated_events_hash",
        "retained_content_hash",
        "retained_block_content_hash",
        "rendered_text_hash",
        "outcome_hash",
    ):
        assert len(manifest[field]) == 64


def test_status_sentinels_have_exactly_one_final_state(tmp_path: Path) -> None:
    status_dir = tmp_path / "outputs/private/run_status/synthetic-job"
    write_running(
        status_dir,
        job_id="synthetic-job",
        started_at="2030-01-01T00:00:00Z",
        git_commit="a" * 40,
        environment_audit="outputs/private/run_status/synthetic-job/environment_audit.json",
        notebook_outputs=("outputs/private/executed/03.ipynb",),
        repository_root=tmp_path,
    )
    complete = write_final(
        status_dir,
        state="COMPLETE",
        ended_at="2030-01-01T00:01:00Z",
        exit_status=0,
        repository_root=tmp_path,
    )
    assert (status_dir / "RUNNING.json").is_file()
    assert complete.is_file()
    assert not (status_dir / "FAILED.json").exists()
    with pytest.raises(ContractError, match="already exists"):
        write_final(
            status_dir,
            state="FAILED",
            ended_at="2030-01-01T00:02:00Z",
            exit_status=1,
            repository_root=tmp_path,
        )


def test_phase1_reviewer_reads_only_allowlisted_aggregate_report(tmp_path: Path) -> None:
    aggregate = tmp_path / "reports/generated/phase1_protocol_review/protocol_alignment.json"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text('{"status": "PASS"}\n', encoding="utf-8")
    assert load_aggregate_report(aggregate, tmp_path) == {"status": "PASS"}
    governed = tmp_path / "outputs/private/cohorts/patient.parquet"
    with pytest.raises(ContractError, match="reports/generated"):
        load_aggregate_report(governed, tmp_path)


def test_supported_duckdb_arrow_reader_is_used() -> None:
    sources = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("src/temporal_ehr/data/history.py", "src/temporal_ehr/features/count_baseline.py")
    )
    assert ".to_arrow_reader(" in sources
    assert ".fetch_record_batch(" not in sources


def test_slurm_alignment_runner_declares_private_caches_and_sentinels() -> None:
    script = Path("scripts/slurm/11_run_phase1_alignment.sbatch").read_text(encoding="utf-8")
    for variable in (
        "IPYTHONDIR",
        "JUPYTER_CONFIG_DIR",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
    ):
        assert f"export {variable}=" in script
    assert "RUNNING" in script and "COMPLETE" in script and "FAILED" in script
    assert "03_phase1_protocol_review.ipynb" in script
    assert "bash -" not in script


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def test_protocol_qc_integration_writes_only_aggregate_and_hash_manifests(tmp_path: Path) -> None:
    prediction = dt.datetime(2024, 1, 1)
    event_rows: list[dict[str, object]] = []
    cohort_rows: list[dict[str, object]] = []
    for index, split in enumerate(("train", "validation"), start=1):
        subject_id = 100 + index
        for days in (240, 125, 61, 10):
            event_rows.append(
                {
                    "subject_id": subject_id,
                    "time": prediction - dt.timedelta(days=days),
                    "code": f"SYN/{days}",
                    "numeric_value": None,
                    "text_value": None,
                    "unit": None,
                    "omop_table": "observation",
                }
            )
        event_rows.append(dict(event_rows[-1]))
        cohort_rows.append(
            {
                "instance_key": f"synthetic-{index}",
                "subject_id": subject_id,
                "prediction_time": prediction,
                "task": "synthetic_task",
                "family": "laboratory_result",
                "split": split,
                "target": bool(index % 2),
            }
        )
    event_path = tmp_path / "events.parquet"
    cohort_path = tmp_path / "cohort.parquet"
    pq.write_table(pa.Table.from_pylist(event_rows), event_path)
    pq.write_table(pa.Table.from_pylist(cohort_rows), cohort_path)

    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    parquet_bytes = io.BytesIO()
    pq.write_table(pa.table({"code": ["SYN/10"], "description": ["Invented event"]}), parquet_bytes)
    with zipfile.ZipFile(asset_root / "metadata.zip", "w") as archive:
        archive.writestr("metadata/codes.parquet", parquet_bytes.getvalue())
    tokenizer_root = tmp_path / "tokenizer"
    tokenizer_root.mkdir()
    (tokenizer_root / "PINNED_REVISION.txt").write_text("synthetic-revision\n", encoding="utf-8")

    experiment_root = tmp_path / "configs/experiments"
    experiment_root.mkdir(parents=True)
    for name, status in (
        ("pilot_three_task", "PILOT"),
        ("proposed_four_task", "PROPOSED"),
        ("locked_panel", "PENDING"),
    ):
        (experiment_root / f"{name}.yaml").write_text(
            f"status: {status}\ntasks: []\n",
            encoding="utf-8",
        )
    registry = {
        "serialization": {
            "family_window_days": {"laboratory_result": 30},
            "c3_perturbation_seeds": [20260815, 20260816, 20260817],
        },
        "tasks": {"synthetic_task": {"family": "laboratory_result"}},
    }
    report = run_protocol_alignment_qc(
        event_path,
        cohort_path,
        asset_root,
        tokenizer=_WhitespaceTokenizer(),
        tokenizer_root=tokenizer_root,
        registry=registry,
        tasks=("synthetic_task",),
        minimum_group_size=1,
        report_path=tmp_path / "reports/generated/phase1_protocol_review/protocol_alignment.json",
        private_manifest_path=tmp_path / "outputs/private/manifests/alignment.jsonl",
        repository_root=tmp_path,
    )
    assert report["status"] == "PASS"
    assert report["protocol_decision"] == "NO_GO"
    assert report["instances_reviewed"] == 2
    assert all(group["exact_duplicates_removed"]["minimum"] == 1 for group in report["duplicate_removal"])
    public_text = (tmp_path / "reports/generated/phase1_protocol_review/protocol_alignment.json").read_text()
    assert "synthetic-1" not in public_text and "subject_id: 101" not in public_text
    manifest_lines = (tmp_path / "outputs/private/manifests/alignment.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 48
