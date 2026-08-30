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


@dataclass(frozen=True)
class FinalM1Authorization:
    """Immutable train/validation authorization for the locked final M1 experiment."""

    experiment_id: str
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    cohort_path: Path
    private_root: Path
    aggregate_root: Path
    token_budget: int
    c3_seeds: tuple[int, ...]
    primary_model_repository: str
    primary_model_directory_name: str
    confirmatory_model_repository: str
    confirmatory_model_directory_name: str
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


def load_final_m1_authorization(path: Path, *, repository_root: Path | None = None) -> FinalM1Authorization:
    """Load Prompt 3's exact final-panel scope and reject any scientific drift."""
    root = (repository_root or Path.cwd()).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ContractError("Final M1 authorization must be a YAML mapping.")
    if config.get("mode") != "FINAL_VALIDATION" or config.get("status") != "LOCKED":
        raise ContractError("Final M1 requires mode FINAL_VALIDATION and status LOCKED.")
    if config.get("experiment_id") != "locked_four_task_m1_v1":
        raise ContractError("Final M1 experiment identity drifted from locked_four_task_m1_v1.")
    tasks = tuple(config.get("tasks", ()))
    if tasks != FINAL_TASKS:
        raise ContractError("Final M1 tasks must exactly match the prespecified four-task panel in order.")
    if config.get("splits") != ["train", "validation"]:
        raise ContractError("Final M1 is restricted to train and validation, in that order.")
    panel = load_experiment_panel(root / str(config.get("panel_config", "")))
    if require_locked_panel(panel) != FINAL_TASKS:
        raise ContractError("Final M1 authorization does not match the separately locked panel.")
    if panel.get("selection_timing") != "before any pilot embedding result":
        raise ContractError("The final panel does not record a prospective pre-pilot lock.")

    private_root = Path(str(config.get("private_root", "")))
    cohort_path = Path(str(config.get("cohort_path", "")))
    aggregate_root = Path(str(config.get("aggregate_root", "")))
    expected_private = Path("outputs/private/final/locked_four_task_m1_v1")
    if private_root != expected_private or not cohort_path.is_relative_to(expected_private):
        raise ContractError("Final M1 artifacts must use only the versioned final private namespace.")
    pilot_root = Path("outputs/private/pilot")
    if private_root.is_relative_to(pilot_root) or pilot_root.is_relative_to(private_root):
        raise ContractError("Final and pilot artifact namespaces must be completely separate.")
    if aggregate_root != Path("reports/generated/final/locked_four_task_m1_v1"):
        raise ContractError("Final M1 aggregate namespace drifted.")

    release = config.get("dataset_release")
    selection = config.get("selection")
    if (
        not isinstance(release, dict)
        or release.get("name") != "EHRSHOT"
        or not isinstance(release.get("release_id"), str)
        or not isinstance(release.get("official_repository_commit"), str)
        or not isinstance(release.get("event_data_sha256"), str)
        or len(str(release["event_data_sha256"])) != 64
    ):
        raise ContractError("Final M1 requires a pinned EHRSHOT dataset/release identity.")
    if (
        not isinstance(selection, dict)
        or int(selection.get("frozen_seed", -1)) != 20260815
        or int(selection.get("maximum_instances_per_patient_task", 0)) != 1
        or selection.get("duplicate_prediction_time_policy")
        != "exclude_entire_group_regardless_of_label_values"
        or set(selection.get("independent_of", ()))
        != {"outcome", "event_history", "record_length", "model_result"}
    ):
        raise ContractError("Final patient-balanced selection contract drifted.")

    models = config.get("models")
    primary = models.get("primary") if isinstance(models, dict) else None
    confirmatory = models.get("confirmatory") if isinstance(models, dict) else None
    expected_models = (
        (primary, "Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B"),
        (confirmatory, "Qwen/Qwen3-Embedding-8B", "Qwen3-Embedding-8B"),
    )
    for model, repository, directory in expected_models:
        if (
            not isinstance(model, dict)
            or model.get("repository") != repository
            or model.get("directory_name") != directory
            or model.get("frozen") is not True
            or model.get("normalize_embeddings") is not True
            or model.get("full_dimension") is not True
            or int(model.get("batch_size", 0)) < 1
        ):
            raise ContractError(f"Final M1 model contract drifted for {repository}.")
    if int(config.get("token_budget", 0)) != 8192:
        raise ContractError("Final M1 token budget must be exactly 8,192 tokens.")
    c3_seeds = tuple(config.get("c3_seeds", ()))
    if c3_seeds != (20260815, 20260816, 20260817):
        raise ContractError("Final M1 C3 seeds drifted from the prespecified three seeds.")
    if tuple(float(value) for value in config.get("regularization_grid", ())) != (0.01, 0.1, 1.0, 10.0):
        raise ContractError("Final M1 logistic regularization grid drifted.")
    if (
        int(config.get("logistic_seed", -1)) != 20260815
        or int(config.get("bootstrap_seed", -1)) != 20260818
        or int(config.get("bootstrap_replicates", 0)) != 2000
    ):
        raise ContractError("Final M1 fitting/bootstrap contract drifted.")

    representations = config.get("representations")
    if (
        not isinstance(representations, dict)
        or representations.get("baselines")
        != ["B1_latest_window_only", "B2_latest_code_occurrence", "B3_code_count", "B4_training_prevalence"]
        or representations.get("temporal_conditions")
        != [
            "C0_order_only",
            "C1_correct_compact_time",
            "C2_reversed_earlier_gaps",
            "C3_shuffled_earlier_time_labels",
        ]
        or representations.get("content_matched_c0_c3") is not True
        or representations.get("perturb_label_indices_only") is not True
    ):
        raise ContractError("Final M1 representation matrix drifted.")
    throughput = config.get("throughput_gate")
    if (
        not isinstance(throughput, dict)
        or float(throughput.get("contingency_multiplier", 0)) != 2.0
        or int(throughput.get("task_job_time_limit_seconds", 0)) != 43200
        or throughput.get("feasibility_inputs_use_outcomes") is not False
        or throughput.get("confirmatory_priority")
        != [
            "C1_correct_compact_time",
            "C3_shuffled_earlier_time_labels",
            "B1_latest_window_only",
            "B2_latest_code_occurrence",
        ]
    ):
        raise ContractError("Final M1 throughput/8B priority contract drifted.")
    instructions = config.get("embedding_instructions")
    outcomes = config.get("outcome_contracts")
    if not isinstance(instructions, dict) or set(instructions) != set(FINAL_TASKS):
        raise ContractError("Final M1 requires exactly one embedding instruction per final task.")
    if not isinstance(outcomes, dict) or set(outcomes) != set(FINAL_TASKS):
        raise ContractError("Final M1 requires exactly one outcome/shortcut contract per task.")
    prohibitions = config.get("prohibitions")
    if not isinstance(prohibitions, dict) or not prohibitions or not all(prohibitions.values()):
        raise ContractError("Every final M1 scope prohibition must be explicit and enabled.")
    preconditions = config.get("preconditions")
    if (
        not isinstance(preconditions, dict)
        or preconditions.get("recovery_job_id") != 408
        or preconditions.get("pilot_protocol_decision") != "PILOT_COMPLETE_FINAL_PANEL_UNCHANGED"
        or preconditions.get("final_panel_frozen_before_pilot_performance") is not True
    ):
        raise ContractError("Final M1 hard-precondition record drifted.")
    return FinalM1Authorization(
        experiment_id="locked_four_task_m1_v1",
        tasks=FINAL_TASKS,
        splits=("train", "validation"),
        cohort_path=root / cohort_path,
        private_root=root / private_root,
        aggregate_root=root / aggregate_root,
        token_budget=8192,
        c3_seeds=c3_seeds,
        primary_model_repository=str(primary["repository"]),
        primary_model_directory_name=str(primary["directory_name"]),
        confirmatory_model_repository=str(confirmatory["repository"]),
        confirmatory_model_directory_name=str(confirmatory["directory_name"]),
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
