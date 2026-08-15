"""Aggregate predictive metrics and a prevalence-only validation reference."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _metric_set(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(targets, probabilities)),
        "auprc": float(average_precision_score(targets, probabilities)),
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(targets, probabilities)),
    }


def validation_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    reference_probability: float,
    minimum_group_size: int = 20,
) -> dict[str, Any]:
    """Return validation estimates, suppressing groups smaller than the privacy threshold."""
    targets = np.asarray(targets, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.shape != probabilities.shape:
        raise ValueError("Targets and probabilities must have identical shapes.")
    if len(targets) < minimum_group_size:
        return {
            "status": "SUPPRESSED",
            "reason": f"group size is below minimum {minimum_group_size}",
        }
    if len(np.unique(targets)) != 2:
        raise ValueError("AUROC and AUPRC require both outcome classes.")
    if not 0.0 <= reference_probability <= 1.0:
        raise ValueError("Reference probability must be between zero and one.")
    reference = np.full(targets.shape, reference_probability, dtype=np.float64)
    return {
        "status": "REPORTED",
        "group_size": len(targets),
        "prevalence": float(targets.mean()),
        "model": _metric_set(targets, probabilities),
        "prevalence_only_reference": {
            "probability_fit_on_training": float(reference_probability),
            **_metric_set(targets, reference),
        },
    }
