"""Shared model and evaluation utilities for the CMPS 355 MDD experiment.

This module does NOT decide:
- preprocessing / normalization
- gene-selection method
- CV structure
- hyperparameter search space

Those decisions are handled separately.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def build_elastic_net_model(
    C: float,
    l1_ratio: float,
    random_state: int = 42,
) -> LogisticRegression:
    """Create the elastic-net logistic-regression classifier."""
    if C <= 0:
        raise ValueError("C must be > 0")
    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be between 0 and 1")

    return LogisticRegression(
        solver="saga",
        C=C,
        l1_ratio=l1_ratio,
        max_iter=10_000,
        random_state=random_state,
    )


def classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the metrics specified in the approved proposal."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    if y_true.shape != probabilities.shape:
        raise ValueError("y_true and probabilities must have the same shape")

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def predict_and_score(
    model: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Generate MDD probabilities and evaluate them."""
    probabilities = model.predict_proba(X)[:, 1]
    return classification_metrics(y, probabilities, threshold)