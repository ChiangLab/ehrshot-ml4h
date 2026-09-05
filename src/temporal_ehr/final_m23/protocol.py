"""Fail-closed Prompt 4 configuration and immutable Prompt 3 gates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.experiments import FINAL_TASKS, load_experiment_panel, require_locked_panel
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.serialization.integrity import stable_hash

M3_SEEDS = (20260815, 20260816, 20260817)
M2_SENSITIVITY = (
    "B1_latest_window_only",
    "C0_order_only",
    "C2_reversed_earlier_gaps",
    "C3_shuffled_earlier_time_labels:seed=20260815",
    "C3_shuffled_earlier_time_labels:seed=20260816",
    "C3_shuffled_earlier_time_labels:seed=20260817",
)


@dataclass(frozen=True)
class FinalM23Authorization:
    """Exact train/validation-only authorization for Prompt 4."""

    experiment_id: str
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    private_root: Path
    aggregate_root: Path
    m1_config_path: Path
    m1_report_path: Path
    m1_report_sha256: str
    prompt3_handoff_path: Path
    prompt3_handoff_sha256: str
    model_repository: str
    model_directory_name: str
    model_batch_size: int
    m2: dict[str, Any]
    m3: dict[str, Any]
    evaluation: dict[str, Any]
    raw: dict[str, Any]


def _exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ContractError(f"{context} keys drifted from the locked Prompt 4 contract.")
    return value


def load_final_m23_authorization(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> FinalM23Authorization:
    """Load Prompt 4 without allowing panel, data-boundary, model, or test drift."""
    root = (repository_root or Path.cwd()).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ContractError("Prompt 4 authorization must be a YAML mapping.")
    if (
        config.get("experiment_id") != "locked_four_task_m2_m3_v1"
        or config.get("mode") != "FINAL_VALIDATION"
        or config.get("status") != "LOCKED"
        or tuple(config.get("tasks", ())) != FINAL_TASKS
        or config.get("splits") != ["train", "validation"]
    ):
        raise ContractError("Prompt 4 identity, panel, or development splits drifted.")
    panel = load_experiment_panel(root / str(config.get("panel_config", "")))
    if require_locked_panel(panel) != FINAL_TASKS:
        raise ContractError("Prompt 4 does not match the prospectively locked four-task panel.")

    upstream = config.get("upstream_m1")
    if not isinstance(upstream, dict) or (
        upstream.get("fit_recovery_job_id") != 451
        or upstream.get("fit_recovery_commit") != "ef1b0debfa963ee365645ec2f754d8d129e071ce"
        or upstream.get("protocol_decision") != "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED"
    ):
        raise ContractError("Prompt 4 requires the exact verified job-451 Prompt 3 handoff.")
    for field in ("report_sha256", "handoff_sha256"):
        value = upstream.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"Prompt 4 upstream {field} is not a SHA-256 digest.")

    private_root = Path(str(config.get("private_root", "")))
    aggregate_root = Path(str(config.get("aggregate_root", "")))
    if private_root != Path("outputs/private/final/locked_four_task_m2_m3_v1"):
        raise ContractError("Prompt 4 private namespace drifted.")
    if aggregate_root != Path("reports/generated/final/locked_four_task_m2_m3_v1"):
        raise ContractError("Prompt 4 aggregate namespace drifted.")

    embedding = config.get("embedding_model")
    if not isinstance(embedding, dict) or any(
        (
            embedding.get("repository") != "Qwen/Qwen3-Embedding-0.6B",
            embedding.get("directory_name") != "Qwen3-Embedding-0.6B",
            embedding.get("frozen") is not True,
            embedding.get("normalize_embeddings_float32") is not True,
            embedding.get("full_dimension") is not True,
            int(embedding.get("batch_size", 0)) < 1,
        )
    ):
        raise ContractError("Prompt 4 frozen Qwen embedding contract drifted.")

    m2 = config.get("m2")
    if not isinstance(m2, dict) or (
        m2.get("approach") != "single_pass_summary_embedding"
        or m2.get("recursive") is not False
        or m2.get("source_history_policy") != "m1_content_matched_8192"
        or m2.get("instruction_model") != "nvidia/Qwen3.6-35B-A3B-NVFP4"
        or m2.get("api_base") != "http://10.45.177.17:8000/v1"
        or m2.get("on_premise_only") is not True
        or m2.get("prompt_version") != "m2_fidelity_v3_grouped_structured"
        or m2.get("reference_condition") != "C1_correct_compact_time"
        or m2.get("fidelity_condition") != "C3_shuffled_earlier_time_labels"
        or int(m2.get("fidelity_seed", -1)) != M3_SEEDS[0]
        or tuple(m2.get("sensitivity_conditions", ())) != M2_SENSITIVITY
    ):
        raise ContractError("Prompt 4 M2 contract drifted.")
    decoding = m2.get("decoding")
    if not isinstance(decoding, dict) or (
        float(decoding.get("temperature", -1)) != 0.0
        or float(decoding.get("top_p", -1)) != 1.0
        or int(decoding.get("seed", -1)) != M3_SEEDS[0]
        or int(decoding.get("max_output_tokens", 0)) < 1
        or decoding.get("chat_template_kwargs") != {"enable_thinking": False}
    ):
        raise ContractError("M2 decoding is not deterministically locked.")
    overflow_decoding = m2.get("overflow_decoding")
    if overflow_decoding != {"max_output_tokens": 8192}:
        raise ContractError("M2 overflow decoding must change only the 8,192-token completion cap.")
    execution = m2.get("execution")
    if not isinstance(execution, dict) or (
        int(execution.get("concurrent_requests", 0)) not in range(1, 17)
        or int(execution.get("request_retries", -1)) not in range(0, 6)
        or int(execution.get("summary_shards_per_task", 0)) != 8
        or int(execution.get("progress_interval_rows", 0)) not in range(1, 101)
    ):
        raise ContractError("M2 execution concurrency, retry, shard, or progress policy is not safely locked.")
    fidelity = m2.get("fidelity")
    if not isinstance(fidelity, dict) or (
        int(fidelity.get("validation_histories", 0)) < 40
        or int(fidelity.get("histories_per_task_condition", 0)) * 8
        != int(fidelity.get("validation_histories", 0))
        or int(fidelity.get("maximum_format_violations", -1)) != 0
    ):
        raise ContractError("M2 fidelity sample is not balanced across four tasks and two conditions.")

    m3 = config.get("m3")
    if not isinstance(m3, dict) or (
        m3.get("approach") != "per_window_embedding_gru"
        or m3.get("reference_condition") != "C1_correct_compact_time"
        or tuple(m3.get("sensitivity_conditions", ())) != M2_SENSITIVITY
        or int(m3.get("max_tokens_per_window", 0)) != 512
        or int(m3.get("embedding_batch_size", 0)) not in (4, 8, 16, 32, 64)
        or int(m3.get("max_recent_nonempty_windows", 0)) != 64
        or m3.get("mean_pool_logistic_baseline") is not True
        or int(m3.get("projection_dimension", 0)) != 256
        or int(m3.get("hidden_size", 0)) != 128
        or float(m3.get("input_dropout", -1)) != 0.1
        or m3.get("optimizer") != "AdamW"
        or m3.get("early_stopping_metric") != "validation_auroc"
        or tuple(m3.get("training_seeds", ())) != M3_SEEDS
    ):
        raise ContractError("Prompt 4 M3 architecture or seed contract drifted.")

    evaluation = config.get("evaluation")
    selection = evaluation.get("approach_selection") if isinstance(evaluation, dict) else None
    if not isinstance(evaluation, dict) or not isinstance(selection, dict) or (
        tuple(float(value) for value in evaluation.get("regularization_grid", ()))
        != (0.01, 0.1, 1.0, 10.0)
        or int(evaluation.get("logistic_seed", -1)) != M3_SEEDS[0]
        or float(selection.get("minimum_margin", -1)) != 0.005
        or selection.get("no_margin_priority") != ["M1", "M2", "M3"]
        or selection.get("second_tiebreaker") != "validation_log_loss"
    ):
        raise ContractError("Prompt 4 evaluation or approach-selection rule drifted.")
    prohibitions = config.get("prohibitions")
    if not isinstance(prohibitions, dict) or not prohibitions or not all(prohibitions.values()):
        raise ContractError("Every Prompt 4 prohibition must be explicit and enabled.")

    return FinalM23Authorization(
        experiment_id=str(config["experiment_id"]),
        tasks=FINAL_TASKS,
        splits=("train", "validation"),
        private_root=root / private_root,
        aggregate_root=root / aggregate_root,
        m1_config_path=root / str(upstream["config"]),
        m1_report_path=root / str(upstream["report"]),
        m1_report_sha256=str(upstream["report_sha256"]),
        prompt3_handoff_path=root / str(upstream["handoff"]),
        prompt3_handoff_sha256=str(upstream["handoff_sha256"]),
        model_repository=str(embedding["repository"]),
        model_directory_name=str(embedding["directory_name"]),
        model_batch_size=int(embedding["batch_size"]),
        m2=dict(m2),
        m3=dict(m3),
        evaluation=dict(evaluation),
        raw=config,
    )


def require_prompt4_hard_gates(
    repository_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Verify Prompt 3 aggregate/handoff hashes and continued held-out lock."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    required = (
        (spec.m1_report_path, spec.m1_report_sha256, "Prompt 3 final M1 report"),
        (spec.prompt3_handoff_path, spec.prompt3_handoff_sha256, "Prompt 3 handoff"),
    )
    for path, expected, label in required:
        if not path.is_file() or file_sha256(path) != expected:
            raise ContractError(f"{label} is missing or hash-mismatched.")
    m1 = json.loads(spec.m1_report_path.read_text(encoding="utf-8"))
    if any(
        (
            m1.get("status") != "PASS",
            m1.get("protocol_decision") != "FINAL_M1_VALIDATION_COMPLETE_TEST_REMAINS_LOCKED",
            m1.get("source_git_commit") != "ef1b0debfa963ee365645ec2f754d8d129e071ce",
            m1.get("test_predictions_or_performance_used") is not False,
            m1.get("test_results_reported") is not False,
            m1.get("m2_run") is not False,
            m1.get("m3_run") is not False,
        )
    ):
        raise ContractError("Prompt 3 final M1 report is not the verified test-locked handoff.")
    handoff = json.loads(spec.prompt3_handoff_path.read_text(encoding="utf-8"))
    serialized = json.dumps(handoff, sort_keys=True)
    if "451" not in serialized or "446" not in serialized or handoff.get("privacy_scope") not in {
        "aggregate-and-metadata-only",
        "aggregate_and_metadata_only",
    }:
        raise ContractError("Prompt 3 handoff lacks recovery provenance or privacy scope.")
    if any(token in serialized.casefold() for token in ("subject_id", "prediction_time", "raw_history")):
        raise ContractError("Prompt 3 handoff contains a prohibited patient-level field.")
    return {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "prompt3_m1_report_sha256": spec.m1_report_sha256,
        "prompt3_handoff_sha256": spec.prompt3_handoff_sha256,
        "prompt3_fit_recovery_job_id": 451,
        "test_predictions_or_performance_used": False,
    }
