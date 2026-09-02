"""Frozen per-window embeddings and deterministic three-seed GRU validation."""

from __future__ import annotations

import copy
import json
import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from temporal_ehr.data.contracts import ContractError, assert_privacy_safe_report
from temporal_ehr.data.history import iter_strict_histories
from temporal_ehr.experiments import load_final_m1_authorization, window_days_for_task
from temporal_ehr.final_m1.embedding import condition_label, condition_slug
from temporal_ehr.final_m1.throughput import read_final_task_texts
from temporal_ehr.final_m23.m2 import _write_parquet_once
from temporal_ehr.final_m23.protocol import FinalM23Authorization, load_final_m23_authorization
from temporal_ehr.pilot.cache import ResumableEmbeddingCache, atomic_json, file_sha256
from temporal_ehr.pilot.evaluation import calibration_summary, metric_set
from temporal_ehr.pilot.model import LocalQwenEncoder, load_model_identity
from temporal_ehr.serialization.blocks import (
    build_prediction_windows,
    deduplicate_events,
    event_from_mapping,
    exact_duplicate_key,
)
from temporal_ehr.serialization.conditions import (
    LabeledBlock,
    TemporalCondition,
    build_full_history_conditions,
)
from temporal_ehr.serialization.integrity import stable_hash
from temporal_ehr.serialization.render import load_code_descriptions, render_labeled_block

BLOCK_LABEL_RE = re.compile(
    r"^\[(?:TIME=\d+_\d+_DAYS_BEFORE_T|CLINICAL_BLOCK|NO_PRE_T_EVENTS)\]$"
)


def split_serialized_windows(text: str, *, maximum_windows: int) -> tuple[str, ...]:
    """Split locked serialization into nonempty chronological blocks and keep the most recent."""
    if maximum_windows < 1:
        raise ValueError("maximum_windows must be positive.")
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if BLOCK_LABEL_RE.fullmatch(stripped):
            current = [stripped]
            blocks.append(current)
        elif current is not None and stripped:
            current.append(line)
    rendered = tuple(
        "\n".join(block).strip()
        for block in blocks
        if len(block) > 1 or block == ["[NO_PRE_T_EVENTS]"]
    )
    if not rendered:
        raise ContractError("M3 received a serialized history with no nonempty observation window.")
    return rendered[-maximum_windows:]


def pad_sequences(
    sequences: Sequence[np.ndarray],
    *,
    maximum_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-pad oldest-to-newest sequences and return lengths plus an explicit mask."""
    if not sequences or maximum_windows < 1:
        raise ValueError("M3 padding requires sequences and a positive maximum.")
    dimension = int(np.asarray(sequences[0]).shape[1])
    output = np.zeros((len(sequences), maximum_windows, dimension), dtype=np.float32)
    lengths = np.empty(len(sequences), dtype=np.int64)
    mask = np.zeros((len(sequences), maximum_windows), dtype=np.bool_)
    for index, sequence in enumerate(sequences):
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != dimension or len(array) < 1:
            raise ContractError("M3 sequence shape or embedding dimension drifted.")
        retained = array[-maximum_windows:]
        lengths[index] = len(retained)
        output[index, : len(retained)] = retained
        mask[index, : len(retained)] = True
    return output, lengths, mask


def fixed_projection(vectors: np.ndarray, *, output_dimension: int, seed: int) -> np.ndarray:
    """Apply one frozen Gaussian projection shared by all rows and model seeds."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim < 2 or output_dimension < 1 or output_dimension > array.shape[-1]:
        raise ContractError("M3 fixed projection dimensions are invalid.")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((array.shape[-1], output_dimension), dtype=np.float32)
    matrix /= np.sqrt(float(output_dimension))
    projected = np.matmul(array, matrix)
    if not np.isfinite(projected).all():
        raise ContractError("M3 fixed projection produced non-finite values.")
    return np.asarray(projected, dtype=np.float32)


def project_then_pad_sequences(
    sequences: Sequence[np.ndarray],
    *,
    maximum_windows: int,
    output_dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project retained windows before padding to avoid a dense 4096-D padded tensor."""
    if not sequences:
        raise ValueError("M3 projection requires at least one sequence.")
    retained = [np.asarray(sequence, dtype=np.float32)[-maximum_windows:] for sequence in sequences]
    if any(array.ndim != 2 or len(array) < 1 for array in retained):
        raise ContractError("M3 projection received an invalid sequence.")
    dimension = int(retained[0].shape[1])
    if any(array.shape[1] != dimension for array in retained):
        raise ContractError("M3 projection received inconsistent embedding dimensions.")
    lengths = [len(array) for array in retained]
    projected = fixed_projection(
        np.concatenate(retained, axis=0),
        output_dimension=output_dimension,
        seed=seed,
    )
    split_at = np.cumsum(lengths[:-1], dtype=np.int64)
    projected_sequences = np.split(projected, split_at)
    return pad_sequences(projected_sequences, maximum_windows=maximum_windows)


def validation_auroc_improved(
    *,
    current_auroc: float,
    best_auroc: float,
    current_log_loss: float,
    best_log_loss: float,
    tolerance: float = 1e-7,
) -> bool:
    """Maximize validation AUROC; use lower log loss only for an AUROC tie."""
    return current_auroc > best_auroc + tolerance or (
        abs(current_auroc - best_auroc) <= tolerance
        and current_log_loss < best_log_loss - tolerance
    )


def m3_stage_labels(spec: FinalM23Authorization, stage: str) -> tuple[str, ...]:
    if stage == "reference":
        return (str(spec.m3["reference_condition"]),)
    if stage != "sensitivity":
        raise ValueError("M3 stage must be reference or sensitivity.")
    path = spec.aggregate_root / "approach_selection.json"
    if not path.is_file():
        raise ContractError("M3 sensitivity requires completed reference approach selection.")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("selected_approach") != "M3" or selection.get("reference_cells_passed") is not True:
        raise ContractError("M3 sensitivity is not authorized by the frozen reference selection.")
    return tuple(str(value) for value in spec.m3["sensitivity_conditions"])


def _condition_for_label(windows: Sequence[Any], label: str, instance_key: str) -> Any:
    if label == "B1_latest_window_only":
        latest = tuple(windows[-1:])
        return TemporalCondition(
            name=label,
            blocks=tuple(LabeledBlock(window, window.window_index) for window in latest),
            eligible=True,
        )
    if ":seed=" in label:
        representation, seed_text = label.split(":seed=", 1)
        seed = int(seed_text)
    else:
        representation = label
        seed = 20260815
    conditions = build_full_history_conditions(windows, seed=seed, instance_key=instance_key)
    if representation not in conditions:
        raise ContractError(f"M3 condition is not a locked temporal representation: {label}")
    return conditions[representation]


def _load_or_build_window_sources_many(
    *,
    repository_root: Path,
    spec: FinalM23Authorization,
    task: str,
    labels: Sequence[str],
    event_path: Path,
    asset_root: Path,
    source_git_commit: str,
) -> dict[str, list[dict[str, Any]]]:
    """Reconstruct every stage condition in one event scan, independent of M1 global packing."""
    labels = tuple(labels)
    if not labels or len(labels) != len(set(labels)):
        raise ContractError("M3 window-source labels must be nonempty and unique.")
    m1 = load_final_m1_authorization(spec.m1_config_path, repository_root=repository_root)
    prepared = read_final_task_texts(m1, task)
    prepared_keys = {
        label: {
            str(row["instance_key"])
            for row in prepared
            if condition_label(row) == label
        }
        for label in labels
    }
    index_path = m1.private_root / "prepared" / task / "patient_window_index.parquet"
    if not index_path.is_file():
        raise ContractError("M3 requires the immutable Prompt 3 patient-window index.")
    index_hash = file_sha256(index_path)
    roots = {
        label: spec.private_root / "m3" / "window_sources" / task / condition_slug(label)
        for label in labels
    }
    input_hashes = {
        label: stable_hash(
            {
                "schema": "m3_complete_window_sources_v1",
                "experiment": spec.experiment_id,
                "source_git_commit": source_git_commit,
                "task": task,
                "condition": label,
                "event_data_sha256": m1.raw["dataset_release"]["event_data_sha256"],
                "cohort": str(m1.cohort_path),
                "patient_window_index_sha256": index_hash,
                "maximum_windows": int(spec.m3["max_recent_nonempty_windows"]),
            }
        )
        for label in labels
    }
    results: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for label in labels:
        rows_path = roots[label] / "rows.parquet"
        complete_path = roots[label] / "COMPLETE.json"
        if complete_path.is_file():
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                not rows_path.is_file()
                or complete.get("input_hash") != input_hashes[label]
                or complete.get("rows_sha256") != file_sha256(rows_path)
            ):
                raise ContractError("Completed M3 window-source cache is stale or hash-mismatched.")
            results[label] = pq.read_table(rows_path).to_pylist()
        else:
            if roots[label].exists() and any(roots[label].iterdir()):
                raise ContractError("Partial M3 window-source cache exists without COMPLETE.")
            missing.append(label)
    if not missing:
        return results

    registry = yaml.safe_load((repository_root / "configs/tasks.yaml").read_text(encoding="utf-8"))
    descriptions = load_code_descriptions(asset_root)
    index_rows = pq.read_table(index_path).to_pylist()
    expected_by_label_instance: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in index_rows:
        row_label = str(row["representation"])
        if row.get("perturbation_seed") is not None:
            row_label = f"{row_label}:seed={row['perturbation_seed']}"
        if row_label in missing:
            expected_by_label_instance.setdefault((row_label, str(row["instance_key"])), []).append(row)
    output: dict[str, list[dict[str, Any]]] = {label: [] for label in missing}
    maximum = int(spec.m3["max_recent_nonempty_windows"])
    for instance in iter_strict_histories(
        event_path,
        m1.cohort_path,
        tasks=(task,),
        splits=m1.splits,
    ):
        events, _ = deduplicate_events(tuple(event_from_mapping(row) for row in instance.events))
        windows = build_prediction_windows(
            events,
            instance.prediction_time,
            window_days=window_days_for_task(task, registry),
        )
        for label in missing:
            condition = _condition_for_label(windows, label, instance.instance_key)
            if not condition.eligible:
                continue
            condition_windows = [item.block for item in condition.blocks]
            actual_hashes = [
                stable_hash(tuple(exact_duplicate_key(event) for event in window.events))
                for window in condition_windows
            ]
            expected = sorted(
                expected_by_label_instance.get((label, instance.instance_key), []),
                key=lambda row: int(row["source_window_index"]),
                reverse=True,
            )
            if [int(row["source_window_index"]) for row in expected] != [
                window.window_index for window in condition_windows
            ] or [str(row["deduplicated_clinical_content_hash"]) for row in expected] != actual_hashes:
                raise ContractError("M3 reconstructed windows drifted from the Prompt 3 patient-window index.")
            rendered = ["\n".join(render_labeled_block(block, descriptions)) for block in condition.blocks]
            retained = rendered[-maximum:] if rendered else ["[NO_PRE_T_EVENTS]"]
            output[label].append(
                {
                    "experiment_id": spec.experiment_id,
                    "task": task,
                    "split": instance.split,
                    "instance_key": instance.instance_key,
                    "target": instance.target,
                    "condition": label,
                    "window_texts": retained,
                    "window_count": len(retained),
                    "source_window_count": len(rendered),
                    "truncated_to_64_windows": len(rendered) > maximum,
                    "source_hash": stable_hash(tuple(retained)),
                }
            )
    for label in missing:
        output[label].sort(key=lambda row: (str(row["split"]), str(row["instance_key"])))
        if {str(row["instance_key"]) for row in output[label]} != prepared_keys[label]:
            raise ContractError("M3 complete-window sources do not align with Prompt 3 eligible patients.")
        rows_path = roots[label] / "rows.parquet"
        rows_hash = _write_parquet_once(rows_path, output[label])
        atomic_json(
            roots[label] / "COMPLETE.json",
            {
                "state": "COMPLETE",
                "experiment_id": spec.experiment_id,
                "task": task,
                "condition": label,
                "input_hash": input_hashes[label],
                "rows": len(output[label]),
                "rows_sha256": rows_hash,
                "complete_history_before_64_window_cap": True,
                "test_split_used": False,
            },
        )
        results[label] = output[label]
    return results


def _sequence_cache(
    *,
    spec: FinalM23Authorization,
    task: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
    instruction: str,
    encoder: LocalQwenEncoder,
    model_artifact_hash: str,
    tokenizer_hash: str,
    source_git_commit: str,
) -> dict[str, Any]:
    maximum = int(spec.m3["max_recent_nonempty_windows"])
    flattened: list[str] = []
    row_keys: list[str] = []
    metadata: list[dict[str, Any]] = []
    for row in rows:
        blocks = tuple(str(value) for value in row["window_texts"])[-maximum:]
        if not blocks:
            raise ContractError("M3 source row contains no observation window or empty-history marker.")
        offset = len(flattened)
        for position, block in enumerate(blocks):
            rendered = f"{instruction.strip()}\n{block}"
            flattened.append(rendered)
            row_keys.append(
                stable_hash(
                    {
                        "schema": "m3_window_embedding_v1",
                        "experiment": spec.experiment_id,
                        "task": task,
                        "condition": label,
                        "instance_key": row["instance_key"],
                        "source_text_hash": row["source_hash"],
                        "window_position": position,
                        "window_text_hash": stable_hash(rendered),
                        "model_artifact_hash": model_artifact_hash,
                        "token_budget": int(spec.m3["max_tokens_per_window"]),
                    }
                )
            )
        metadata.append(
            {
                "experiment_id": spec.experiment_id,
                "task": task,
                "split": row["split"],
                "instance_key": row["instance_key"],
                "target": row["target"],
                "condition": label,
                "sequence_row_index": len(metadata),
                "window_embedding_offset": offset,
                "window_count": len(blocks),
                "source_text_hash": row["source_hash"],
                "maximum_recent_nonempty_windows": maximum,
                "mirrors_sequence_rows": True,
            }
        )
    if not row_keys or len(row_keys) != len(set(row_keys)):
        raise ContractError("M3 flattened window cache keys are empty or duplicated.")
    root = spec.private_root / "m3" / "window_embeddings" / task / condition_slug(label)
    configuration_hash = stable_hash(
        {
            "schema": "m3_window_embedding_cache_v1",
            "experiment_config_hash": stable_hash(spec.raw),
            "source_git_commit": source_git_commit,
            "task": task,
            "condition": label,
            "model_artifact_hash": model_artifact_hash,
            "tokenizer_hash": tokenizer_hash,
            "max_tokens_per_window": int(spec.m3["max_tokens_per_window"]),
            "maximum_windows": maximum,
            "row_keys": row_keys,
        }
    )
    metadata_hash = _write_parquet_once(root / "sequence_metadata.parquet", metadata)
    cache = ResumableEmbeddingCache(
        root,
        row_keys=row_keys,
        dimension=encoder.dimension,
        configuration_hash=configuration_hash,
    )
    pending = cache.pending_indices()
    if not cache.complete_path.is_file():
        cache.begin_attempt()
        batch_size = int(spec.m3["embedding_batch_size"])
        for offset in range(0, len(pending), batch_size):
            indices = pending[offset : offset + batch_size]
            cache.write_rows(
                indices.tolist(),
                encoder.encode([flattened[int(index)] for index in indices], batch_size=batch_size),
            )
        cache.finalize()
        cache.finish_attempt()
    cache.load_complete()
    return {
        "condition": label,
        "histories": len(rows),
        "window_embeddings": len(row_keys),
        "maximum_recent_nonempty_windows": maximum,
        "max_tokens_per_window": int(spec.m3["max_tokens_per_window"]),
        "configuration_hash": configuration_hash,
        "metadata_sha256": metadata_hash,
        "cache_complete_sha256": file_sha256(cache.complete_path),
        "rows_computed_this_attempt": len(pending),
        "cache_reused_without_encoding": len(pending) == 0,
    }


def load_m3_sequences(
    spec: FinalM23Authorization,
    *,
    task: str,
    label: str,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    root = spec.private_root / "m3" / "window_embeddings" / task / condition_slug(label)
    metadata_path = root / "sequence_metadata.parquet"
    complete_path = root / "COMPLETE.json"
    vectors_path = root / "embeddings.npy"
    if not metadata_path.is_file() or not complete_path.is_file() or not vectors_path.is_file():
        raise ContractError(f"M3 sequence cache is incomplete for {task}/{label}.")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("embeddings_sha256") != file_sha256(vectors_path):
        raise ContractError("M3 completed sequence embedding cache hash drifted.")
    metadata = pq.read_table(metadata_path).to_pylist()
    metadata.sort(key=lambda row: int(row["sequence_row_index"]))
    if [int(row["sequence_row_index"]) for row in metadata] != list(range(len(metadata))):
        raise ContractError("M3 sequence metadata row order is not contiguous.")
    vectors = np.load(vectors_path, mmap_mode="r")
    sequences = []
    expected_offset = 0
    for row in metadata:
        offset = int(row["window_embedding_offset"])
        count = int(row["window_count"])
        if offset != expected_offset or count < 1:
            raise ContractError("M3 sequence metadata does not mirror flattened embedding rows.")
        sequences.append(np.asarray(vectors[offset : offset + count], dtype=np.float32))
        expected_offset += count
    if expected_offset != len(vectors):
        raise ContractError("M3 sequence metadata does not cover every embedding row.")
    return metadata, sequences


def _train_gru(
    *,
    train_sequences: Sequence[np.ndarray],
    train_targets: np.ndarray,
    validation_sequences: Sequence[np.ndarray],
    validation_targets: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    try:
        import torch
        from torch import nn
        from torch.nn.utils.rnn import pack_padded_sequence
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - NVIDIA container dependency
        raise RuntimeError("M3 GRU requires PyTorch inside the NVIDIA container.") from exc
    if len(np.unique(train_targets)) != 2 or len(np.unique(validation_targets)) != 2:
        raise ContractError("M3 GRU requires both classes in train and validation.")
    maximum = int(config["max_recent_nonempty_windows"])
    projection_dimension = int(config["projection_dimension"])
    train_x, train_lengths, train_mask = project_then_pad_sequences(
        train_sequences,
        maximum_windows=maximum,
        output_dimension=projection_dimension,
        seed=int(config["projection_seed"]),
    )
    validation_x, validation_lengths, validation_mask = project_then_pad_sequences(
        validation_sequences,
        maximum_windows=maximum,
        output_dimension=projection_dimension,
        seed=int(config["projection_seed"]),
    )
    if not np.array_equal(train_mask.sum(axis=1), train_lengths) or not np.array_equal(
        validation_mask.sum(axis=1), validation_lengths
    ):
        raise ContractError("M3 sequence padding mask does not match lengths.")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SequenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dropout = nn.Dropout(float(config["input_dropout"]))
            self.gru = nn.GRU(
                input_size=projection_dimension,
                hidden_size=int(config["hidden_size"]),
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.output = nn.Linear(int(config["hidden_size"]), 1)

        def forward(self, values: Any, lengths: Any) -> Any:
            dropped = self.input_dropout(values)
            packed = pack_padded_sequence(
                dropped,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.gru(packed)
            return self.output(hidden[-1]).squeeze(1)

    model = SequenceModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    train_dataset = TensorDataset(
        torch.from_numpy(train_x),
        torch.from_numpy(train_lengths),
        torch.from_numpy(np.asarray(train_targets, dtype=np.float32)),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_values = torch.from_numpy(validation_x).to(device)
    validation_length_tensor = torch.from_numpy(validation_lengths).to(device)
    validation_target_tensor = torch.from_numpy(np.asarray(validation_targets, dtype=np.float32)).to(device)
    best_auroc = -math.inf
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    stale = 0
    epochs_run = 0
    for epoch in range(int(config["maximum_epochs"])):
        model.train()
        for values, lengths, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device), lengths.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_values, validation_length_tensor)
            validation_loss = float(criterion(validation_logits, validation_target_tensor).item())
            validation_probability = torch.sigmoid(validation_logits).cpu().numpy()
            validation_auroc = float(metric_set(validation_targets, validation_probability)["auroc"])
        epochs_run = epoch + 1
        improved = validation_auroc_improved(
            current_auroc=validation_auroc,
            best_auroc=best_auroc,
            current_log_loss=validation_loss,
            best_log_loss=best_loss,
        )
        if improved:
            best_auroc = validation_auroc
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    if best_state is None:
        raise ContractError("M3 validation-only early stopping did not retain a model.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(validation_values, validation_length_tensor)).cpu().numpy()
    report = {
        "seed": seed,
        "architecture": "one-layer unidirectional GRU",
        "projection_dimension": projection_dimension,
        "projection_fixed": True,
        "hidden_size": int(config["hidden_size"]),
        "input_dropout": float(config["input_dropout"]),
        "optimizer": "AdamW",
        "validation_only_early_stopping": True,
        "early_stopping_metric": "validation_auroc",
        "epochs_run": epochs_run,
        "best_epoch_zero_based": best_epoch,
        "best_validation_auroc": best_auroc,
        "validation_log_loss_at_best_auroc": best_loss,
        "metrics": metric_set(validation_targets, probability),
        "calibration": calibration_summary(validation_targets, probability),
        "device": str(device),
    }
    return probability, report, best_state


def _fit_one_condition(
    *,
    spec: FinalM23Authorization,
    task: str,
    label: str,
    seed: int,
) -> dict[str, Any]:
    output_root = spec.private_root / "m3" / "fits" / task / condition_slug(label) / f"seed_{seed}"
    predictions_path = output_root / "validation_predictions.parquet"
    model_path = output_root / "model.pt"
    fit_report_path = output_root / "FIT_REPORT.json"
    complete_path = output_root / "COMPLETE.json"
    if complete_path.is_file():
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if any(not path.is_file() for path in (predictions_path, model_path, fit_report_path)) or (
            complete.get("predictions_sha256") != file_sha256(predictions_path)
            or complete.get("model_sha256") != file_sha256(model_path)
            or complete.get("fit_report_sha256") != file_sha256(fit_report_path)
        ):
            raise ContractError("Completed M3 fit artifacts are missing or hash-mismatched.")
        return json.loads(fit_report_path.read_text(encoding="utf-8"))
    if output_root.exists() and any(output_root.iterdir()):
        raise ContractError("Partial M3 fit exists without COMPLETE; refusing silent reuse or overwrite.")
    metadata, sequences = load_m3_sequences(spec, task=task, label=label)
    train_indices = [index for index, row in enumerate(metadata) if row["split"] == "train"]
    validation_indices = [index for index, row in enumerate(metadata) if row["split"] == "validation"]
    if not train_indices or not validation_indices:
        raise ContractError("M3 condition lacks aligned train/validation rows.")
    train_targets = np.asarray([metadata[index]["target"] for index in train_indices], dtype=np.int8)
    validation_targets = np.asarray([metadata[index]["target"] for index in validation_indices], dtype=np.int8)
    probability, report, state = _train_gru(
        train_sequences=[sequences[index] for index in train_indices],
        train_targets=train_targets,
        validation_sequences=[sequences[index] for index in validation_indices],
        validation_targets=validation_targets,
        config=spec.m3,
        seed=seed,
    )
    predictions = [
        {
            "task": task,
            "condition": label,
            "seed": seed,
            "instance_key": metadata[index]["instance_key"],
            "target": bool(metadata[index]["target"]),
            "probability": float(value),
        }
        for index, value in zip(validation_indices, probability, strict=True)
    ]
    predictions_hash = _write_parquet_once(predictions_path, predictions)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("M3 model serialization requires PyTorch in the NVIDIA container.") from exc
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_model = output_root / ".model.pt.tmp"
    torch.save(state, temporary_model)
    temporary_model.replace(model_path)
    model_hash = file_sha256(model_path)
    final_report = {**report, "predictions_sha256": predictions_hash, "model_sha256": model_hash}
    atomic_json(fit_report_path, final_report)
    complete = {
        "state": "COMPLETE",
        "experiment_id": spec.experiment_id,
        "task": task,
        "condition": label,
        "seed": seed,
        "predictions_sha256": predictions_hash,
        "model_sha256": model_hash,
        "fit_report_sha256": file_sha256(fit_report_path),
    }
    atomic_json(complete_path, complete)
    return final_report


def prepare_m3_task(
    *,
    repository_root: Path,
    config_path: Path,
    model_root: Path,
    event_path: Path,
    asset_root: Path,
    task: str,
    stage: str,
    source_git_commit: str,
) -> dict[str, Any]:
    """Build immutable task/condition window caches once, independently of training seed."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    if task not in spec.tasks:
        raise ContractError("M3 task is outside the locked mapping.")
    m1 = load_final_m1_authorization(spec.m1_config_path, repository_root=repository_root)
    labels = m3_stage_labels(spec, stage)
    identity = load_model_identity(
        model_root / spec.model_directory_name,
        expected_repository=spec.model_repository,
    )
    cache_reports = []
    started = time.perf_counter()
    sources = _load_or_build_window_sources_many(
        repository_root=repository_root,
        spec=spec,
        task=task,
        labels=labels,
        event_path=event_path,
        asset_root=asset_root,
        source_git_commit=source_git_commit,
    )
    encoder = LocalQwenEncoder(identity, max_seq_length=int(spec.m3["max_tokens_per_window"]))
    for label in labels:
        rows = sources[label]
        if not rows:
            raise ContractError(f"M3 prepared source lacks {task}/{label}.")
        cache_reports.append(
            _sequence_cache(
                spec=spec,
                task=task,
                label=label,
                rows=rows,
                instruction=m1.instructions[task],
                encoder=encoder,
                model_artifact_hash=identity.artifact_hash,
                tokenizer_hash=identity.tokenizer_hash,
                source_git_commit=source_git_commit,
            )
        )
    report = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "stage": stage,
        "task": task,
        "embedding_model": identity.repository,
        "embedding_model_artifact_hash": identity.artifact_hash,
        "frozen_window_encoder": True,
        "window_cache_reports": cache_reports,
        "runtime_seconds": time.perf_counter() - started,
        "patient_alignment_passed": True,
        "test_split_used": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    report_path = spec.aggregate_root / "m3" / "tasks" / f"{task}_{stage}_prepare.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.is_file():
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise ContractError("Refusing to overwrite a different M3 task/seed aggregate.")
    else:
        report_path.write_text(text, encoding="utf-8")
    return report


def run_m3_task_seed(
    *,
    repository_root: Path,
    config_path: Path,
    task: str,
    seed: int,
    stage: str,
    source_git_commit: str,
) -> dict[str, Any]:
    """Train one deterministic seed from immutable seed-independent window caches."""
    spec = load_final_m23_authorization(config_path, repository_root=repository_root)
    if task not in spec.tasks or seed not in tuple(spec.m3["training_seeds"]):
        raise ContractError("M3 task or seed is outside the locked mapping.")
    labels = m3_stage_labels(spec, stage)
    prepare_path = spec.aggregate_root / "m3" / "tasks" / f"{task}_{stage}_prepare.json"
    if not prepare_path.is_file():
        raise ContractError("M3 seed training requires the completed task preparation report.")
    preparation = json.loads(prepare_path.read_text(encoding="utf-8"))
    if (
        preparation.get("status") != "PASS"
        or preparation.get("source_git_commit") != source_git_commit
        or preparation.get("stage") != stage
        or preparation.get("task") != task
        or preparation.get("patient_alignment_passed") is not True
        or preparation.get("test_split_used") is not False
        or {row.get("condition") for row in preparation.get("window_cache_reports", [])}
        != set(labels)
    ):
        raise ContractError("M3 task preparation provenance or condition coverage drifted.")
    started = time.perf_counter()
    fit_reports = {
        label: _fit_one_condition(spec=spec, task=task, label=label, seed=seed)
        for label in labels
    }
    report = {
        "status": "PASS",
        "experiment_id": spec.experiment_id,
        "source_git_commit": source_git_commit,
        "stage": stage,
        "task": task,
        "seed": seed,
        "embedding_model": preparation["embedding_model"],
        "embedding_model_artifact_hash": preparation["embedding_model_artifact_hash"],
        "frozen_window_encoder": True,
        "seed_independent_cache_prepared_once": True,
        "preparation_report_sha256": file_sha256(prepare_path),
        "window_cache_reports": preparation["window_cache_reports"],
        "fits": fit_reports,
        "runtime_seconds": time.perf_counter() - started,
        "patient_alignment_passed": True,
        "test_split_used": False,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    assert_privacy_safe_report(text)
    report_path = spec.aggregate_root / "m3" / "tasks" / f"{task}_seed_{seed}_{stage}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.is_file():
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise ContractError("Refusing to overwrite a different M3 task/seed aggregate.")
    else:
        report_path.write_text(text, encoding="utf-8")
    return report
