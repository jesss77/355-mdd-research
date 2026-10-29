"""Generic cross-validation mechanics for the CMPS 355 MDD experiment.

This module intentionally does NOT choose:
- the CV splitter or number of folds
- preprocessing / normalization
- gene-selection method
- hyperparameter grid

Pass those choices in later. Any estimator supplied here is cloned and fitted
inside each training fold only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from src.modeling import classification_metrics


METRICS = ("auroc", "f1", "sensitivity", "specificity")


@dataclass(frozen=True)
class CVResult:
    fold_metrics: tuple[dict[str, float], ...]
    summary: dict[str, dict[str, float]]


def _summarize(folds: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}

    for metric in METRICS:
        values = np.asarray([fold[metric] for fold in folds], dtype=float)
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    return summary


def run_cross_validation(estimator, X: np.ndarray, y: np.ndarray, cv) -> CVResult:
    """Fit/evaluate a supplied estimator independently inside each CV fold."""
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D samples x features matrix")
    if y.ndim != 1:
        raise ValueError("y must be a 1D label vector")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("y must contain both binary classes 0 and 1")

    folds: list[dict[str, float]] = []

    for fold_number, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        if np.intersect1d(train_idx, valid_idx).size:
            raise ValueError(f"Fold {fold_number}: training/validation overlap")

        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])

        probabilities = model.predict_proba(X[valid_idx])[:, 1]
        metrics = classification_metrics(y[valid_idx], probabilities)

        folds.append({
            "fold": float(fold_number),
            "train_n": float(len(train_idx)),
            "valid_n": float(len(valid_idx)),
            **metrics,
        })

    if not folds:
        raise ValueError("CV splitter produced no folds")

    return CVResult(
        fold_metrics=tuple(folds),
        summary=_summarize(folds),
    )
