"""Fail-closed experiment-panel configuration and governance gates."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError

PANEL_STATUSES = ("PILOT", "PROPOSED", "PENDING", "LOCKED")


def load_experiment_panel(path: Path) -> dict[str, Any]:
    """Load and validate an experiment-panel document without choosing a panel."""
    with path.open(encoding="utf-8") as handle:
        panel = yaml.safe_load(handle)
    if not isinstance(panel, dict):
        raise ContractError("Experiment panel must be a YAML mapping.")
    status = panel.get("status")
    tasks = panel.get("tasks")
    if status not in PANEL_STATUSES:
        raise ContractError(f"Experiment panel status must be one of {PANEL_STATUSES}.")
    if not isinstance(tasks, list) or any(not isinstance(task, str) or not task for task in tasks):
        raise ContractError("Experiment panel tasks must be a list of nonempty task names.")
    if len(tasks) != len(set(tasks)):
        raise ContractError("Experiment panel contains duplicate task names.")
    if status == "LOCKED" and not tasks:
        raise ContractError("A LOCKED experiment panel must contain at least one task.")
    return panel


def require_locked_panel(panel: Mapping[str, Any]) -> tuple[str, ...]:
    """Return locked tasks or stop downstream embedding/model training."""
    if panel.get("status") != "LOCKED":
        raise ContractError(
            "Downstream embedding training is blocked: configs/experiments/locked_panel.yaml "
            "must be reviewed with Jeff and have status LOCKED."
        )
    tasks = panel.get("tasks")
    if not isinstance(tasks, list) or not tasks or any(not isinstance(task, str) or not task for task in tasks):
        raise ContractError("The LOCKED panel must contain a nonempty list of task names.")
    if len(tasks) != len(set(tasks)):
        raise ContractError("The LOCKED panel contains duplicate task names.")
    return tuple(tasks)


def window_days_for_task(task: str, registry: Mapping[str, Any]) -> int:
    """Resolve the configured prediction-relative window from task family."""
    try:
        family = str(registry["tasks"][task]["family"])
        window_days = int(registry["serialization"]["family_window_days"][family])
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"No configured serialization window for task {task!r}.") from exc
    if window_days not in (30, 90):
        raise ContractError(f"Unsupported primary serialization window for task {task!r}: {window_days}.")
    return window_days
