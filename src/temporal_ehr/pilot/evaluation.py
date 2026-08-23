"""Deterministic pilot readouts, calibration summaries, and paired intervals."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from temporal_ehr.data.contracts import ContractError


def metric_set(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape or len(np.unique(targets)) != 2:
        raise ContractError("Metric inputs must be aligned and contain both classes.")
    return {
        "auroc": float(roc_auc_score(targets, probabilities)),
        "auprc": float(average_precision_score(targets, probabilities)),
        "brier_score": float(brier_score_loss(targets, probabilities)),
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
    }


def calibration_summary(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_class_count: int = 20,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    counts = np.bincount(targets, minlength=2)
    if counts.min() < minimum_class_count:
        return {"status": "SUPPRESSED", "reason": "fewer than 20 observations in one outcome class"}
    if np.ptp(probabilities) < 1e-8:
        return {"status": "SUPPRESSED", "reason": "constant predictions do not identify calibration slope"}
    logits = np.log(np.clip(probabilities, 1e-6, 1 - 1e-6) / np.clip(1 - probabilities, 1e-6, 1))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(logits[:, None], targets)
    return {
        "status": "REPORTED",
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def fit_logistic_readout(
    train_embeddings: np.ndarray,
    train_targets: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_targets: np.ndarray,
    *,
    c_grid: Sequence[float],
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use one deterministic validation-selection procedure for every representation."""
    if train_embeddings.shape[0] != len(train_targets) or validation_embeddings.shape[0] != len(validation_targets):
        raise ContractError("Embedding and label rows are misaligned.")
    if len(np.unique(train_targets)) != 2 or len(np.unique(validation_targets)) != 2:
        raise ContractError("Logistic readout requires both outcome classes in train and validation.")
    candidates: list[dict[str, float]] = []
    models: dict[float, LogisticRegression] = {}
    for value in c_grid:
        if value <= 0:
            raise ValueError("Regularization values must be positive.")
        model = LogisticRegression(C=float(value), solver="liblinear", max_iter=2000, random_state=random_seed)
        model.fit(train_embeddings, train_targets)
        probability = model.predict_proba(validation_embeddings)[:, 1]
        candidates.append(
            {"C": float(value), "validation_auroc": float(roc_auc_score(validation_targets, probability))}
        )
        models[float(value)] = model
    selected = min(candidates, key=lambda row: (-row["validation_auroc"], row["C"]))
    probability = models[selected["C"]].predict_proba(validation_embeddings)[:, 1]
    return probability, {
        "selected_C": selected["C"],
        "candidate_results": candidates,
        "tie_rule": "highest validation AUROC; smallest C wins an exact tie",
        "metrics": metric_set(validation_targets, probability),
        "calibration": calibration_summary(validation_targets, probability),
    }


def paired_bootstrap_interval(
    targets: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired one-instance-per-patient AUROC difference interval."""
    targets = np.asarray(targets, dtype=np.int8)
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if targets.shape != first.shape or targets.shape != second.shape:
        raise ContractError("Paired bootstrap inputs are not aligned.")
    observed = float(roc_auc_score(targets, first) - roc_auc_score(targets, second))
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(replicates):
        indices = rng.integers(0, len(targets), size=len(targets))
        sampled_targets = targets[indices]
        if len(np.unique(sampled_targets)) != 2:
            continue
        estimates.append(
            float(
                roc_auc_score(sampled_targets, first[indices])
                - roc_auc_score(sampled_targets, second[indices])
            )
        )
    if len(estimates) < max(20, replicates // 2):
        raise ContractError("Too few valid paired bootstrap replicates.")
    lower, upper = np.percentile(np.asarray(estimates), [2.5, 97.5])
    return {
        "metric": "paired validation AUROC difference",
        "estimate": observed,
        "ci_95": [float(lower), float(upper)],
        "requested_replicates": replicates,
        "valid_replicates": len(estimates),
        "resampling_unit": "patient (one selected instance per patient/task)",
    }
