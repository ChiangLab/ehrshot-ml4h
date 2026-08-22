from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import pytest

import scripts.build_handoff_bundle as handoff
import temporal_ehr.run_status as run_status
from temporal_ehr.alignment_recovery import validate_recovery_artifacts, write_validation_report
from temporal_ehr.data.contracts import ContractError

SOURCE_JOB_ID = "407"
SOURCE_COMMIT = "f9a370c7fe1cb13edd126128cfddc79100852913"
RECOVERY_COMMIT = "b" * 40
STARTED = "2026-08-20T16:42:02Z"
ENDED = "2026-08-20T19:06:03Z"
REPORT_MTIME = "2026-08-20T19:05:58Z"
NOTEBOOK_MTIME = "2026-08-20T19:05:59Z"


def _epoch(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _notebook(*, incomplete: bool = False, error: bool = False) -> dict[str, object]:
    outputs: list[dict[str, object]] = []
    if error:
        outputs.append(
            {
                "output_type": "error",
                "ename": "SyntheticError",
                "evalue": "synthetic failure",
                "traceback": [],
            }
        )
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None if incomplete else 1,
                "id": "synthetic-cell",
                "metadata": {},
                "outputs": outputs,
                "source": ["value = 1"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _recovery_fixture(
    root: Path,
    *,
    incomplete: bool = False,
    error: bool = False,
    report_mtime: str = REPORT_MTIME,
) -> dict[str, Path]:
    report = root / "reports/generated/phase1_protocol_review/protocol_alignment.json"
    notebook = root / "outputs/private/executed_notebooks/03_phase1_protocol_review.ipynb"
    failed = root / "outputs/private/run_status/407/FAILED.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    notebook.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"status": "PASS", "instances_reviewed": 3}\n', encoding="utf-8")
    notebook.write_text(json.dumps(_notebook(incomplete=incomplete, error=error)), encoding="utf-8")
    failed.write_text(
        json.dumps(
            {
                "state": "FAILED",
                "job_id": SOURCE_JOB_ID,
                "git_commit": SOURCE_COMMIT,
                "started_at": STARTED,
                "ended_at": ENDED,
                "exit_status": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(report, (_epoch(report_mtime), _epoch(report_mtime)))
    os.utime(notebook, (_epoch(NOTEBOOK_MTIME), _epoch(NOTEBOOK_MTIME)))
    return {"report": report, "notebook": notebook, "failed": failed}


def _validate(root: Path, paths: dict[str, Path], *, report_mtime: str = REPORT_MTIME) -> dict[str, object]:
    return validate_recovery_artifacts(
        repository_root=root,
        report_path=paths["report"],
        notebook_path=paths["notebook"],
        source_failed_sentinel=paths["failed"],
        source_job_id=SOURCE_JOB_ID,
        source_git_commit=SOURCE_COMMIT,
        recovery_git_commit=RECOVERY_COMMIT,
        source_started_at=STARTED,
        source_ended_at=ENDED,
        expected_report_size=paths["report"].stat().st_size,
        expected_notebook_size=paths["notebook"].stat().st_size,
        expected_report_mtime=report_mtime,
        expected_notebook_mtime=NOTEBOOK_MTIME,
    )


def test_recovery_rejects_missing_and_stale_outputs(tmp_path: Path) -> None:
    paths = _recovery_fixture(tmp_path)
    paths["report"].unlink()
    with pytest.raises(ContractError, match="missing"):
        validate_recovery_artifacts(
            repository_root=tmp_path,
            report_path=paths["report"],
            notebook_path=paths["notebook"],
            source_failed_sentinel=paths["failed"],
            source_job_id=SOURCE_JOB_ID,
            source_git_commit=SOURCE_COMMIT,
            recovery_git_commit=RECOVERY_COMMIT,
            source_started_at=STARTED,
            source_ended_at=ENDED,
            expected_report_size=1,
            expected_notebook_size=paths["notebook"].stat().st_size,
            expected_report_mtime=REPORT_MTIME,
            expected_notebook_mtime=NOTEBOOK_MTIME,
        )

    stale = "2026-08-20T15:05:58Z"
    paths = _recovery_fixture(tmp_path, report_mtime=stale)
    with pytest.raises(ContractError, match="stale or outside"):
        _validate(tmp_path, paths, report_mtime=stale)


@pytest.mark.parametrize(
    ("incomplete", "error", "message"),
    [(True, False, "incomplete_code_cells"), (False, True, "error_cells")],
)
def test_recovery_rejects_incomplete_or_error_notebooks(
    tmp_path: Path, incomplete: bool, error: bool, message: str
) -> None:
    paths = _recovery_fixture(tmp_path, incomplete=incomplete, error=error)
    with pytest.raises(RuntimeError, match=message):
        _validate(tmp_path, paths)


def test_recovery_records_hashes_without_overwriting_source_failure(tmp_path: Path) -> None:
    paths = _recovery_fixture(tmp_path)
    before = paths["failed"].read_bytes()
    payload = _validate(tmp_path, paths)
    output = tmp_path / "outputs/private/run_status/999/recovery_validation.json"
    write_validation_report(output, payload, tmp_path)

    assert paths["failed"].read_bytes() == before
    assert sorted(path.name for path in paths["failed"].parent.iterdir()) == ["FAILED.json"]
    assert payload["source_failed_sentinel"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert all(len(artifact["sha256"]) == 64 for artifact in payload["artifacts"].values())
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


def test_containerized_alignment_command_construction() -> None:
    main = Path("scripts/slurm/11_run_phase1_alignment.sbatch").read_text(encoding="utf-8")
    for command in (
        "scripts/audit_environment.py",
        "scripts/execute_notebook.py",
        "scripts/validate_executed_notebook.py",
        "scripts/review_phase1_results.py",
    ):
        assert f'run_in_container "${{PYTHON}}" {command}' in main
    assert 'PYTHONPATH=src "${PYTHON}" scripts/' not in main

    recovery = Path("scripts/slurm/11b_recover_phase1_alignment.sbatch").read_text(encoding="utf-8")
    assert 'SOURCE_ALIGNMENT_JOB_ID}" != "407"' in recovery
    assert 'run_in_container "${PYTHON}" scripts/validate_phase1_alignment_recovery.py' in recovery
    assert 'run_in_container "${PYTHON}" scripts/review_phase1_results.py' in recovery
    assert "execute_notebook.py" not in recovery
    assert "outputs/private/run_status/407/FAILED.json" not in recovery


def test_handoff_bundle_discards_disallowed_aggregate_and_private_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in handoff.HASHED_PROTOCOL_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic protocol\n", encoding="utf-8")
    report = tmp_path / handoff.AGGREGATE_REPORTS[0]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "status": "PASS",
                "protocol_decision": "NO_GO",
                "instances_reviewed": 12,
                "panel_status": {"locked_panel": "PENDING"},
                "checklist": {"content_matching_passed": True},
                "subject_id": 987654,
                "predictions": [0.12345],
                "raw_history": "PRIVATE-HISTORY-MARKER",
            }
        ),
        encoding="utf-8",
    )
    status_dir = tmp_path / "outputs/private/run_status/999"
    status_dir.mkdir(parents=True)
    (status_dir / "COMPLETE.json").write_text(
        json.dumps(
            {
                "state": "COMPLETE",
                "job_id": "999",
                "git_commit": "b" * 40,
                "exit_status": 0,
                "notebook_outputs": ["PRIVATE-NOTEBOOK-MARKER"],
            }
        ),
        encoding="utf-8",
    )
    poison = tmp_path / "outputs/private/manifests/private_rows.json"
    poison.parent.mkdir(parents=True)
    poison.write_text("PRIVATE-MANIFEST-ROW-MARKER", encoding="utf-8")
    monkeypatch.setattr(
        handoff,
        "_git_metadata",
        lambda root: {"head_commit": "a" * 40, "working_tree_dirty": True},
    )

    payload = handoff.collect_handoff(tmp_path, ("999",))
    output = tmp_path / "reports/generated/handoff/project_handoff.json"
    handoff.write_handoff(output, payload, tmp_path)
    text = output.read_text(encoding="utf-8")
    assert "987654" not in text
    assert "0.12345" not in text
    assert "PRIVATE-HISTORY-MARKER" not in text
    assert "PRIVATE-NOTEBOOK-MARKER" not in text
    assert "PRIVATE-MANIFEST-ROW-MARKER" not in text
    handoff.assert_handoff_payload_safe(payload)


def test_atomic_final_status_leaves_no_partial_file_on_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_dir = tmp_path / "outputs/private/run_status/synthetic"
    run_status.write_running(
        status_dir,
        job_id="synthetic",
        started_at="2030-01-01T00:00:00Z",
        git_commit="a" * 40,
        environment_audit="",
        notebook_outputs=(),
        repository_root=tmp_path,
    )
    running_before = (status_dir / "RUNNING.json").read_bytes()

    def fail_dump(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("synthetic write failure")

    monkeypatch.setattr(run_status.json, "dump", fail_dump)
    with pytest.raises(OSError, match="synthetic write failure"):
        run_status.write_final(
            status_dir,
            state="COMPLETE",
            ended_at="2030-01-01T00:01:00Z",
            exit_status=0,
            repository_root=tmp_path,
        )
    assert (status_dir / "RUNNING.json").read_bytes() == running_before
    assert not (status_dir / "COMPLETE.json").exists()
    assert list(status_dir.glob(".COMPLETE.json.*")) == []


def test_final_status_rejects_inconsistent_state_and_exit_code(tmp_path: Path) -> None:
    status_dir = tmp_path / "outputs/private/run_status/synthetic"
    run_status.write_running(
        status_dir,
        job_id="synthetic",
        started_at="2030-01-01T00:00:00Z",
        git_commit="a" * 40,
        environment_audit="",
        notebook_outputs=(),
        repository_root=tmp_path,
    )
    with pytest.raises(ContractError, match="COMPLETE requires"):
        run_status.write_final(
            status_dir,
            state="COMPLETE",
            ended_at="2030-01-01T00:01:00Z",
            exit_status=1,
            repository_root=tmp_path,
        )
    assert not (status_dir / "COMPLETE.json").exists()
