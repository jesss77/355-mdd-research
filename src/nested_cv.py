"""Nested cross-validation for the CMPS 355 MDD gene-level experiment.

Design frozen before model fitting:
- development cohort: GSE98793 only
- outer CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- inner CV: StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
- feature selection: top 50 genes by ANOVA F score, fitted inside CV
- scaling: StandardScaler, fitted inside CV
- classifier: elastic-net logistic regression with solver="saga"
- inner selection metric: AUROC
- C grid: [0.01, 0.1, 1, 10, 100]
- l1_ratio grid: [0.1, 0.5, 0.9]
- classification threshold: 0.5

External cohorts are never loaded or used here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.gene_features import build_gene_selector
from src.model_data import DEVELOPMENT, CohortData, load_cohort
from src.modeling import classification_metrics
from src.pipeline_factory import build_model_pipeline


OUTER_SPLITS = 5
INNER_SPLITS = 5
OUTER_RANDOM_STATE = 42
INNER_RANDOM_STATE = 43

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
L1_RATIO_GRID = [0.1, 0.5, 0.9]
CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class NestedFoldResult:
    fold: int
    train_n: int
    valid_n: int
    best_inner_auroc: float
    best_C: float
    best_l1_ratio: float
    auroc: float
    f1: float
    sensitivity: float
    specificity: float
    selected_gene_ids: tuple[str, ...]
    selected_gene_symbols: tuple[str, ...]


@dataclass(frozen=True)
class NestedCVResult:
    folds: tuple[NestedFoldResult, ...]
    summary: dict[str, dict[str, float]]


def build_gene_pipeline():
    """Construct the leakage-safe gene-level pipeline."""
    return build_model_pipeline(
        selector=build_gene_selector(k=50),
    )


def build_parameter_grid() -> dict[str, list[float]]:
    """Return the frozen elastic-net search grid."""
    return {
        "model__C": C_GRID,
        "model__l1_ratio": L1_RATIO_GRID,
    }


def _summarize(
    fold_results: list[NestedFoldResult],
) -> dict[str, dict[str, float]]:
    metrics = ("auroc", "f1", "sensitivity", "specificity")
    summary: dict[str, dict[str, float]] = {}

    for metric in metrics:
        values = np.asarray(
            [getattr(fold, metric) for fold in fold_results],
            dtype=float,
        )
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }

    return summary


def run_nested_gene_cv(data: CohortData) -> NestedCVResult:
    """Run the frozen nested-CV procedure on GSE98793 only."""
    if data.cohort != DEVELOPMENT:
        raise ValueError(
            "Nested model selection is restricted to GSE98793 development data."
        )

    X = np.asarray(data.X)
    y = np.asarray(data.y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D samples x genes matrix")
    if y.ndim != 1:
        raise ValueError("y must be a 1D label vector")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if X.shape[1] != len(data.gene_ids):
        raise ValueError("Gene metadata does not match X columns")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both binary classes 0 and 1 are required")

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    fold_results: list[NestedFoldResult] = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=INNER_RANDOM_STATE,
        )

        search = GridSearchCV(
            estimator=build_gene_pipeline(),
            param_grid=build_parameter_grid(),
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        probabilities = best_model.predict_proba(X_valid)[:, 1]

        metrics = classification_metrics(
            y_valid,
            probabilities,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        selector = best_model.named_steps["select"]
        selected_mask = selector.get_support()
        selected_indices = np.flatnonzero(selected_mask)

        if len(selected_indices) != 50:
            raise AssertionError(
                f"Outer fold {fold_number}: expected 50 selected genes, "
                f"got {len(selected_indices)}"
            )

        selected_gene_ids = tuple(
            data.gene_ids[index] for index in selected_indices
        )
        selected_gene_symbols = tuple(
            data.gene_symbols[index] for index in selected_indices
        )

        fold_results.append(
            NestedFoldResult(
                fold=fold_number,
                train_n=len(train_idx),
                valid_n=len(valid_idx),
                best_inner_auroc=float(search.best_score_),
                best_C=float(search.best_params_["model__C"]),
                best_l1_ratio=float(
                    search.best_params_["model__l1_ratio"]
                ),
                auroc=metrics["auroc"],
                f1=metrics["f1"],
                sensitivity=metrics["sensitivity"],
                specificity=metrics["specificity"],
                selected_gene_ids=selected_gene_ids,
                selected_gene_symbols=selected_gene_symbols,
            )
        )

    return NestedCVResult(
        folds=tuple(fold_results),
        summary=_summarize(fold_results),
    )


def print_nested_cv_result(result: NestedCVResult) -> None:
    """Print a compact reproducible summary of the nested-CV run."""
    print("Nested gene-level CV: GSE98793 only")
    print(
        f"outer={OUTER_SPLITS} folds, inner={INNER_SPLITS} folds, "
        f"grid={len(C_GRID) * len(L1_RATIO_GRID)} combinations"
    )

    for fold in result.folds:
        print(
            f"fold {fold.fold}: "
            f"train={fold.train_n}, valid={fold.valid_n}, "
            f"best C={fold.best_C:g}, "
            f"best l1_ratio={fold.best_l1_ratio:g}, "
            f"inner AUROC={fold.best_inner_auroc:.3f}, "
            f"outer AUROC={fold.auroc:.3f}, "
            f"F1={fold.f1:.3f}, "
            f"sensitivity={fold.sensitivity:.3f}, "
            f"specificity={fold.specificity:.3f}"
        )

    print("\nOuter-fold summary:")
    for metric, values in result.summary.items():
        print(
            f"{metric}: "
            f"{values['mean']:.3f} ± {values['std']:.3f}"
        )


def main() -> None:
    data = load_cohort(DEVELOPMENT)
    result = run_nested_gene_cv(data)
    print_nested_cv_result(result)


if __name__ == "__main__":
    main()
