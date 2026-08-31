"""Outcome-blind 0.6B/8B throughput gate and confirmatory feasibility plan."""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.experiments import FinalM1Authorization, load_final_m1_authorization
from temporal_ehr.final_m1.foundation import _write_immutable_json
from temporal_ehr.pilot.cache import file_sha256
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity
from temporal_ehr.serialization.integrity import stable_hash


def validate_foundation_reuse(
    *,
    repository_root: Path,
    spec: FinalM1Authorization,
    foundation: Mapping[str, Any],
    current_source_git_commit: str,
) -> dict[str, Any]:
    """Verify exact prior-foundation provenance across one execution-only config change."""
    current_config_hash = stable_hash(spec.raw)
    if (
        foundation.get("source_git_commit") == current_source_git_commit
        and foundation.get("experiment_config_hash") == current_config_hash
    ):
        return {
            "mode": "same_source",
            "foundation_job_id": None,
            "foundation_source_git_commit": current_source_git_commit,
            "foundation_experiment_config_hash": current_config_hash,
            "current_source_git_commit": current_source_git_commit,
            "current_experiment_config_hash": current_config_hash,
            "execution_only_change": None,
        }

    reuse = spec.raw.get("foundation_reuse")
    if not isinstance(reuse, dict):
        raise ContractError("Final foundation does not match current source and has no reuse authorization.")
    source_job_id = str(reuse.get("source_job_id", ""))
    source_commit = str(reuse.get("source_git_commit", ""))
    source_config_hash = str(reuse.get("source_experiment_config_hash", ""))
    change = reuse.get("permitted_execution_only_change")
    if not isinstance(change, dict):
        raise ContractError("Final foundation reuse lacks one exact execution-only change.")
    if (
        foundation.get("status") != "PASS"
        or foundation.get("protocol_decision") != "FINAL_FOUNDATION_COMPLETE_M1_AUTHORIZED"
        or foundation.get("source_git_commit") != source_commit
        or foundation.get("experiment_config_hash") != source_config_hash
        or foundation.get("test_predictions_or_performance_used") is not False
    ):
        raise ContractError("Reusable final foundation identity or no-test gate failed.")

    status_path = repository_root / f"outputs/private/run_status/{source_job_id}/COMPLETE.json"
    if not status_path.is_file():
        raise ContractError("Reusable final foundation lacks its job-specific COMPLETE sentinel.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("state") != "COMPLETE"
        or status.get("exit_status") != 0
        or str(status.get("job_id")) != source_job_id
        or status.get("git_commit") != source_commit
        or status.get("run_kind") != "final_foundation"
    ):
        raise ContractError("Reusable final foundation job sentinel failed provenance checks.")

    reconstructed = deepcopy(spec.raw)
    reconstructed.pop("foundation_reuse", None)
    field = change.get("field")
    if field != "throughput_gate.minimum_runtime_reserve_seconds":
        raise ContractError("Only the throughput runtime reserve may differ for foundation reuse.")
    current_value = int(reconstructed["throughput_gate"]["minimum_runtime_reserve_seconds"])
    if current_value != int(change.get("current_value", -1)):
        raise ContractError("Current throughput reserve does not match its reuse authorization.")
    reconstructed["throughput_gate"]["minimum_runtime_reserve_seconds"] = int(
        change.get("previous_value", -1)
    )
    if stable_hash(reconstructed) != source_config_hash:
        raise ContractError("Current configuration differs from the foundation by more than the authorized reserve.")
    return {
        "mode": "validated_execution_only_reuse",
        "foundation_job_id": source_job_id,
        "foundation_source_git_commit": source_commit,
        "foundation_experiment_config_hash": source_config_hash,
        "current_source_git_commit": current_source_git_commit,
        "current_experiment_config_hash": current_config_hash,
        "execution_only_change": {
            "field": field,
            "previous_value": int(change["previous_value"]),
            "current_value": int(change["current_value"]),
        },
        "cohort_or_prepared_artifact_changed": False,
        "test_predictions_or_performance_used": False,
    }


def read_final_task_texts(spec: FinalM1Authorization, task: str) -> list[dict[str, Any]]:
    task_root = spec.private_root / "prepared" / task
    complete_path = task_root / "COMPLETE.json"
    texts_path = task_root / "texts.parquet"
    metadata_path = task_root / "patient_window_metadata.parquet"
    if not complete_path.is_file() or not texts_path.is_file() or not metadata_path.is_file():
        raise ContractError(f"Final preparation is incomplete for task {task}.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        complete.get("texts_sha256") != file_sha256(texts_path)
        or complete.get("patient_window_metadata_sha256") != file_sha256(metadata_path)
    ):
        raise ContractError(f"Final prepared inputs or metadata drifted for task {task}.")
    rows = pq.read_table(texts_path).to_pylist()
    metadata = pq.read_table(metadata_path).to_pylist()
    rows.sort(key=lambda row: int(row["embedding_row_index"]))
    metadata.sort(key=lambda row: int(row["embedding_row_index"]))
    if [int(row["embedding_row_index"]) for row in rows] != list(range(len(rows))):
        raise ContractError("Final prepared embedding row indices are not contiguous.")
    if [row["cache_key"] for row in metadata] != [row["cache_key"] for row in rows]:
        raise ContractError("Patient-window metadata does not mirror every final prepared embedding row.")
    return rows


def _condition_label(row: Mapping[str, Any]) -> str:
    seed = row.get("perturbation_seed")
    return str(row["representation"]) if seed is None else f"{row['representation']}:seed={seed}"


def _priority_rows(rows: Sequence[Mapping[str, Any]], priority: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if row["representation"] == priority]


def plan_confirmatory_cells(
    *,
    row_counts_by_task: Mapping[str, Mapping[str, int]],
    primary_documents_per_second: float,
    confirmatory_documents_per_second: Mapping[str, float],
    priorities: Sequence[str],
    contingency_multiplier: float,
    task_job_time_limit_seconds: int,
    minimum_runtime_reserve_seconds: int,
) -> dict[str, Any]:
    """Greedily admit 8B cells in prespecified order using runtime only."""
    if primary_documents_per_second <= 0 or contingency_multiplier != 2.0:
        raise ContractError("Throughput feasibility requires positive primary speed and a 2x contingency.")
    tasks: dict[str, Any] = {}
    primary_feasible = True
    for task, counts in row_counts_by_task.items():
        primary_rows = int(counts.get("__primary_total__", 0))
        if primary_rows < 1:
            raise ContractError(f"No primary embedding rows were counted for {task}.")
        primary_seconds = contingency_multiplier * primary_rows / primary_documents_per_second
        available = task_job_time_limit_seconds - minimum_runtime_reserve_seconds - primary_seconds
        selected: list[str] = []
        estimates: list[dict[str, Any]] = []
        if available < 0:
            primary_feasible = False
        else:
            used = 0.0
            for priority in priorities:
                speed = float(confirmatory_documents_per_second.get(priority, 0.0))
                rows = int(counts.get(priority, 0))
                seconds = contingency_multiplier * rows / speed if speed > 0 and rows > 0 else float("inf")
                fits = used + seconds <= available
                estimates.append(
                    {
                        "priority": priority,
                        "rows": rows,
                        "contingent_runtime_seconds": seconds if seconds != float("inf") else None,
                        "selected": fits,
                    }
                )
                if not fits:
                    break
                selected.append(priority)
                used += seconds
        tasks[task] = {
            "primary_rows": primary_rows,
            "primary_contingent_runtime_seconds": primary_seconds,
            "task_job_time_limit_seconds": task_job_time_limit_seconds,
            "minimum_runtime_reserve_seconds": minimum_runtime_reserve_seconds,
            "confirmatory_available_seconds": max(0.0, available),
            "confirmatory_priorities_selected": selected,
            "confirmatory_estimates": estimates,
            "decision_inputs": "runtime, row counts, fixed priority, and memory only; no outcomes or performance",
        }
    return {
        "primary_feasible_with_2x_contingency": primary_feasible,
        "tasks": tasks,
        "confirmatory_priority": list(priorities),
        "contingency_multiplier": contingency_multiplier,
        "outcome_performance_used_for_feasibility": False,
    }


def _benchmark_encoder(
    *,
    encoder: LocalQwenEncoder,
    batch_size: int,
    candidates: Sequence[Mapping[str, Any]],
    warmup: bool,
) -> dict[str, Any]:
    if warmup:
        encoder.encode(["Synthetic warm-up record."], batch_size=1)
    encoder.reset_peak_memory()
    started = time.perf_counter()
    encoder.encode([str(row["text"]) for row in candidates], batch_size=batch_size)
    elapsed = time.perf_counter() - started
    result = {
        "status": "PASS",
        "sample_count": len(candidates),
        "batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "documents_per_second": len(candidates) / elapsed,
        "tokens_per_second": sum(int(row["token_count"]) for row in candidates) / elapsed,
        "peak_gpu_memory_bytes": encoder.peak_memory_bytes(),
        "device_name": encoder.device_name(),
    }
    return result


def benchmark_final_throughput(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    source_git_commit: str,
) -> dict[str, Any]:
    """Benchmark prepared inputs without reading labels or outcome performance."""
    spec = load_final_m1_authorization(config_path, repository_root=repository_root)
    foundation_path = spec.aggregate_root / "foundation.json"
    if not foundation_path.is_file():
        raise ContractError("Final throughput requires a completed foundation aggregate.")
    foundation = json.loads(foundation_path.read_text(encoding="utf-8"))
    foundation_provenance = validate_foundation_reuse(
        repository_root=repository_root,
        spec=spec,
        foundation=foundation,
        current_source_git_commit=source_git_commit,
    )
    task_rows = {task: read_final_task_texts(spec, task) for task in spec.tasks}
    all_rows = [row for rows in task_rows.values() for row in rows]
    # One deterministic near-limit input from every concrete representation/seed/task cell.
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for task, rows in task_rows.items():
        for row in rows:
            grouped[(task, _condition_label(row))].append(row)
    primary_candidates = [
        max(rows, key=lambda row: (int(row["token_count"]), str(row["text_hash"])))
        for rows in grouped.values()
    ]
    primary_identity = load_model_identity(
        model_root / spec.primary_model_directory_name,
        expected_repository=spec.primary_model_repository,
    )
    primary_encoder = LocalQwenEncoder(primary_identity, max_seq_length=spec.token_budget)
    primary_result = _benchmark_encoder(
        encoder=primary_encoder,
        batch_size=int(spec.raw["models"]["primary"]["batch_size"]),
        candidates=primary_candidates,
        warmup=True,
    )
    del primary_encoder
    gc.collect()

    priorities = tuple(spec.raw["throughput_gate"]["confirmatory_priority"])
    confirmatory_identity = load_model_identity(
        model_root / spec.confirmatory_model_directory_name,
        expected_repository=spec.confirmatory_model_repository,
    )
    confirmatory_by_priority: dict[str, Any] = {}
    confirmatory_speeds: dict[str, float] = {}
    confirmatory_encoder = LocalQwenEncoder(confirmatory_identity, max_seq_length=spec.token_budget)
    for priority_index, priority in enumerate(priorities):
        candidates = []
        for task, rows in task_rows.items():
            eligible = _priority_rows(rows, priority)
            if not eligible:
                raise ContractError(f"No prepared final input is available for 8B priority {priority}/{task}.")
            candidates.append(max(eligible, key=lambda row: (int(row["token_count"]), str(row["text_hash"]))))
        result = _benchmark_encoder(
            encoder=confirmatory_encoder,
            batch_size=int(spec.raw["models"]["confirmatory"]["batch_size"]),
            candidates=candidates,
            warmup=priority_index == 0,
        )
        confirmatory_by_priority[priority] = result
        confirmatory_speeds[priority] = float(result["documents_per_second"])
    del confirmatory_encoder
    gc.collect()

    row_counts: dict[str, dict[str, int]] = {}
    for task, rows in task_rows.items():
        priority_counts: defaultdict[str, int] = defaultdict(int)
        for row in rows:
            priority_counts[str(row["representation"])] += 1
        row_counts[task] = {
            "__primary_total__": len(rows),
            **{key: int(value) for key, value in priority_counts.items()},
        }
    plan = plan_confirmatory_cells(
        row_counts_by_task=row_counts,
        primary_documents_per_second=float(primary_result["documents_per_second"]),
        confirmatory_documents_per_second=confirmatory_speeds,
        priorities=priorities,
        contingency_multiplier=float(spec.raw["throughput_gate"]["contingency_multiplier"]),
        task_job_time_limit_seconds=int(spec.raw["throughput_gate"]["task_job_time_limit_seconds"]),
        minimum_runtime_reserve_seconds=int(spec.raw["throughput_gate"]["minimum_runtime_reserve_seconds"]),
    )
    if plan["primary_feasible_with_2x_contingency"] is not True:
        raise ContractError("Primary 0.6B M1 does not fit the 12-hour task jobs with a 2x contingency.")
    report = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "experiment_config_hash": stable_hash(spec.raw),
        "source_git_commit": source_git_commit,
        "foundation_provenance": foundation_provenance,
        "token_budget": spec.token_budget,
        "sample_policy": "one deterministic near-limit prepared input per task and concrete condition",
        "warmup_policy": "one fully synthetic input per model excluded from timing",
        "patient_text_printed": False,
        "outcomes_or_performance_read": False,
        "primary_model": {
            "repository": primary_identity.repository,
            "revision_hash": stable_hash(primary_identity.revision),
            "artifact_hash": primary_identity.artifact_hash,
            "tokenizer_hash": primary_identity.tokenizer_hash,
            **primary_result,
        },
        "confirmatory_model": {
            "repository": confirmatory_identity.repository,
            "revision_hash": stable_hash(confirmatory_identity.revision),
            "artifact_hash": confirmatory_identity.artifact_hash,
            "tokenizer_hash": confirmatory_identity.tokenizer_hash,
            "benchmarks_by_priority": confirmatory_by_priority,
        },
        "feasibility_plan": plan,
        "prepared_embedding_rows": len(all_rows),
        "test_predictions_or_performance_used": False,
    }
    text = json.dumps(report, sort_keys=True)
    assert_privacy_safe_report(text)
    _write_immutable_json(spec.aggregate_root / "throughput.json", report, aggregate=True)
    return report
