"""Atomic, job-scoped Phase 1 alignment status sentinels."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.serialization.integrity import require_private_output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def write_running(
    status_dir: Path,
    *,
    job_id: str,
    started_at: str,
    git_commit: str,
    environment_audit: str,
    notebook_outputs: Sequence[str],
    repository_root: Path | None = None,
) -> Path:
    """Create RUNNING exactly once for a new private job status directory."""
    directory = require_private_output(status_dir, repository_root)
    sentinels = tuple(directory.glob("*.json")) if directory.exists() else ()
    if sentinels:
        raise ContractError("Refusing to overwrite an existing job status sentinel.")
    destination = directory / "RUNNING.json"
    _atomic_json(
        destination,
        {
            "state": "RUNNING",
            "job_id": str(job_id),
            "started_at": started_at,
            "ended_at": None,
            "git_commit": git_commit,
            "environment_audit": environment_audit,
            "notebook_outputs": list(notebook_outputs),
            "exit_status": None,
        },
    )
    return destination


def write_final(
    status_dir: Path,
    *,
    state: str,
    ended_at: str,
    exit_status: int,
    repository_root: Path | None = None,
) -> Path:
    """Create exactly one COMPLETE or FAILED sentinel after RUNNING."""
    if state not in {"COMPLETE", "FAILED"}:
        raise ValueError("Final state must be COMPLETE or FAILED.")
    directory = require_private_output(status_dir, repository_root)
    running_path = directory / "RUNNING.json"
    if not running_path.is_file():
        raise ContractError("Cannot finalize a job without RUNNING.json.")
    if (directory / "COMPLETE.json").exists() or (directory / "FAILED.json").exists():
        raise ContractError("A final job status sentinel already exists.")
    payload = json.loads(running_path.read_text(encoding="utf-8"))
    payload.update({"state": state, "ended_at": ended_at, "exit_status": int(exit_status)})
    destination = directory / f"{state}.json"
    _atomic_json(destination, payload)
    return destination
