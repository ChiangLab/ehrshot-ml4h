"""Fail-closed experiment-panel configuration and governance gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError

PANEL_STATUSES = ("PILOT", "PROPOSED", "PENDING", "LOCKED")
PILOT_TASKS = ("lab_anemia", "new_acutemi", "guo_los")
FINAL_TASKS = ("lab_anemia", "guo_readmission", "new_hypertension", "chexpert")


@dataclass(frozen=True)
class PilotAuthorization:
    """Explicit, isolated authorization for one engineering pilot."""

    experiment_id: str
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    cohort_path: Path
    private_root: Path
    aggregate_root: Path
    token_budget: int
    c3_seeds: tuple[int, ...]
    model_repository: str
    model_directory_name: str
    instructions: dict[str, str]
    raw: dict[str, Any]


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


def load_pilot_authorization(path: Path, *, repository_root: Path | None = None) -> PilotAuthorization:
    """Load an explicit PILOT config without weakening the final-panel gate."""
    root = (repository_root or Path.cwd()).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("mode") != "PILOT" or config.get("status") != "PILOT":
        raise ContractError("Pilot execution requires an explicit mode: PILOT and status: PILOT authorization.")
    experiment_id = config.get("experiment_id")
    tasks = config.get("tasks")
    splits = config.get("splits")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ContractError("Pilot authorization requires a nonempty experiment_id.")
    if not isinstance(tasks, list) or not tasks or len(tasks) != len(set(tasks)):
        raise ContractError("Pilot authorization requires unique tasks.")
    if tuple(tasks) != PILOT_TASKS or config.get("approach") != "M1_direct_serialization":
        raise ContractError("Pilot authorization is restricted to the exact three-task M1 engineering pilot.")
    if splits != ["train", "validation"]:
        raise ContractError("Pilot authorization is restricted to train and validation, in that order.")
    if Path(str(config.get("cohort_path", ""))) != Path("outputs/private/cohorts/primary_panel.parquet"):
        raise ContractError("Pilot authorization requires the governed three-task primary-panel cohort.")
    panel = load_experiment_panel(root / str(config.get("panel_config", "")))
    if panel.get("status") != "PILOT" or tuple(panel["tasks"]) != tuple(tasks):
        raise ContractError("Pilot authorization tasks do not exactly match the PILOT panel configuration.")
    final_panel = load_experiment_panel(root / str(config.get("locked_final_panel_config", "")))
    if require_locked_panel(final_panel) != FINAL_TASKS:
        raise ContractError("The separately locked final panel drifted from the prespecified four tasks.")
    private_root = Path(str(config.get("private_root", "")))
    aggregate_root = Path(str(config.get("aggregate_root", "")))
    final_namespace = Path(str(config.get("artifact_namespaces", {}).get("final", "")))
    pilot_namespace = Path(str(config.get("artifact_namespaces", {}).get("pilot", "")))
    expected_private = Path("outputs/private/pilot") / experiment_id
    if private_root != expected_private or pilot_namespace != expected_private:
        raise ContractError("Pilot artifacts must use the experiment-specific outputs/private/pilot namespace.")
    if (
        final_namespace != Path("outputs/private/final/locked_four_task_panel_v1")
        or final_namespace == pilot_namespace
        or final_namespace.is_relative_to(pilot_namespace)
        or pilot_namespace.is_relative_to(final_namespace)
    ):
        raise ContractError("Pilot and final artifact namespaces must be completely separate.")
    if not str(aggregate_root).startswith(f"reports/generated/pilot/{experiment_id}"):
        raise ContractError("Pilot aggregate output must use its experiment-specific reports/generated namespace.")
    model = config.get("model")
    instructions = config.get("embedding_instructions")
    if (
        not isinstance(model, dict)
        or model.get("frozen") is not True
        or model.get("normalize_embeddings") is not True
        or model.get("repository") != "Qwen/Qwen3-Embedding-0.6B"
        or model.get("directory_name") != "Qwen3-Embedding-0.6B"
        or model.get("revision_file") != "PINNED_REVISION.txt"
        or model.get("artifact_manifest") != "MODEL_ARTIFACT_MANIFEST.json"
    ):
        raise ContractError("Pilot encoder must be the exact frozen, pinned, normalized Qwen 0.6B model.")
    if not isinstance(instructions, dict) or set(instructions) != set(tasks):
        raise ContractError("Pilot authorization requires exactly one embedding instruction per task.")
    c3_seeds = tuple(config.get("c3_seeds", ()))
    if c3_seeds != (20260815, 20260816, 20260817):
        raise ContractError("Pilot C3 seeds must match the three prespecified seeds.")
    if int(config.get("token_budget", 0)) != 4096:
        raise ContractError("The authorized engineering pilot token budget is exactly 4096.")
    if tuple(float(value) for value in config.get("regularization_grid", ())) != (0.01, 0.1, 1.0, 10.0):
        raise ContractError("Pilot logistic regularization grid drifted from the prespecified procedure.")
    if (
        int(config.get("logistic_seed", -1)) != 20260815
        or int(config.get("bootstrap_seed", -1)) != 20260818
        or int(config.get("bootstrap_replicates", 0)) != 2000
    ):
        raise ContractError("Pilot fitting or paired-bootstrap procedure drifted from its prespecification.")
    prohibitions = config.get("prohibitions")
    if not isinstance(prohibitions, dict) or not all(prohibitions.values()):
        raise ContractError("Every pilot scope prohibition must be explicit and enabled.")
    return PilotAuthorization(
        experiment_id=experiment_id,
        tasks=tuple(str(task) for task in tasks),
        splits=("train", "validation"),
        cohort_path=root / str(config["cohort_path"]),
        private_root=root / private_root,
        aggregate_root=root / aggregate_root,
        token_budget=4096,
        c3_seeds=c3_seeds,
        model_repository=str(model["repository"]),
        model_directory_name=str(model["directory_name"]),
        instructions={str(key): str(value) for key, value in instructions.items()},
        raw=config,
    )


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
