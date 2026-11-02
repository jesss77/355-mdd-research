"""All-shared-genes secondary baseline for the CMPS 355 MDD experiment.

This baseline uses all 16,467 shared genes rather than selecting the top 50.

Design:
- development cohort: GSE98793 only
- no feature-selection step
- StandardScaler fitted inside training folds
- elastic-net logistic regression
- outer CV: stratified 5-fold
- inner CV: stratified 5-fold
- inner selection metric: AUROC
- same frozen elastic-net grid as the primary gene model
- classification threshold: 0.5

External cohorts are never loaded or used here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.model_data import DEVELOPMENT, ROOT, CohortData, load_cohort
from src.modeling import classification_metrics
from src.nested_cv import (
    C_GRID,
    CLASSIFICATION_THRESHOLD,
    INNER_RANDOM_STATE,
    INNER_SPLITS,
    L1_RATIO_GRID,
    OUTER_RANDOM_STATE,
    OUTER_SPLITS,
    build_parameter_grid,
)
from src.pipeline_factory import build_model_pipeline


DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "all_genes_baseline"
)


@dataclass(frozen=True)
class BaselineFoldResult:
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


@dataclass(frozen=True)
class BaselineCVResult:
    folds: tuple[BaselineFoldResult, ...]
    summary: dict[str, dict[str, float]]


@dataclass(frozen=True)
class FinalBaselineResult:
    model: Pipeline
    best_inner_auroc: float
    best_C: float
    best_l1_ratio: float


def build_baseline_pipeline() -> Pipeline:
    """Build the all-shared-genes pipeline."""
    return build_model_pipeline(
        selector="passthrough",
    )


def _summarize(
    fold_results: list[BaselineFoldResult],
) -> dict[str, dict[str, float]]:
    metrics = (
        "auroc",
        "f1",
        "sensitivity",
        "specificity",
    )

    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for metric in metrics:
        values = np.asarray(
            [
                getattr(fold, metric)
                for fold in fold_results
            ],
            dtype=float,
        )

        summary[metric] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        }

    return summary


def _validate_development_data(
    data: CohortData,
) -> tuple[np.ndarray, np.ndarray]:
    if data.cohort != DEVELOPMENT:
        raise ValueError(
            "Baseline fitting is restricted "
            "to GSE98793 development data."
        )

    X = np.asarray(
        data.X,
        dtype=float,
    )

    y = np.asarray(
        data.y,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D samples x genes matrix"
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a 1D label vector"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain "
            "the same number of samples"
        )

    if X.shape[1] != len(
        data.gene_ids
    ):
        raise ValueError(
            "Gene metadata does not "
            "match X columns"
        )

    if set(np.unique(y)) != {
        0,
        1,
    }:
        raise ValueError(
            "Both binary classes are required"
        )

    return X, y


def run_nested_baseline_cv(
    data: CohortData,
) -> BaselineCVResult:
    """Estimate baseline performance with nested CV."""
    X, y = _validate_development_data(
        data
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    fold_results: list[
        BaselineFoldResult
    ] = []

    for fold_number, (
        train_idx,
        valid_idx,
    ) in enumerate(
        outer_cv.split(
            X,
            y,
        ),
        start=1,
    ):
        X_train = X[
            train_idx
        ]

        y_train = y[
            train_idx
        ]

        X_valid = X[
            valid_idx
        ]

        y_valid = y[
            valid_idx
        ]

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=(
                INNER_RANDOM_STATE
            ),
        )

        search = GridSearchCV(
            estimator=(
                build_baseline_pipeline()
            ),
            param_grid=(
                build_parameter_grid()
            ),
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        search.fit(
            X_train,
            y_train,
        )

        best_model = (
            search.best_estimator_
        )

        probabilities = (
            best_model.predict_proba(
                X_valid
            )[:, 1]
        )

        metrics = (
            classification_metrics(
                y_valid,
                probabilities,
                threshold=(
                    CLASSIFICATION_THRESHOLD
                ),
            )
        )

        fold_results.append(
            BaselineFoldResult(
                fold=fold_number,
                train_n=len(
                    train_idx
                ),
                valid_n=len(
                    valid_idx
                ),
                best_inner_auroc=float(
                    search.best_score_
                ),
                best_C=float(
                    search.best_params_[
                        "model__C"
                    ]
                ),
                best_l1_ratio=float(
                    search.best_params_[
                        "model__l1_ratio"
                    ]
                ),
                auroc=(
                    metrics["auroc"]
                ),
                f1=(
                    metrics["f1"]
                ),
                sensitivity=(
                    metrics[
                        "sensitivity"
                    ]
                ),
                specificity=(
                    metrics[
                        "specificity"
                    ]
                ),
            )
        )

    return BaselineCVResult(
        folds=tuple(
            fold_results
        ),
        summary=_summarize(
            fold_results
        ),
    )


def fit_final_baseline(
    data: CohortData,
) -> FinalBaselineResult:
    """Fit the final all-genes baseline on GSE98793."""
    X, y = _validate_development_data(
        data
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=(
            INNER_RANDOM_STATE
        ),
    )

    search = GridSearchCV(
        estimator=(
            build_baseline_pipeline()
        ),
        param_grid=(
            build_parameter_grid()
        ),
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )

    search.fit(
        X,
        y,
    )

    return FinalBaselineResult(
        model=(
            search.best_estimator_
        ),
        best_inner_auroc=float(
            search.best_score_
        ),
        best_C=float(
            search.best_params_[
                "model__C"
            ]
        ),
        best_l1_ratio=float(
            search.best_params_[
                "model__l1_ratio"
            ]
        ),
    )


def _write_fold_metrics(
    result: BaselineCVResult,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "fold_metrics.tsv"
    )

    fieldnames = [
        "fold",
        "train_n",
        "valid_n",
        "best_inner_auroc",
        "best_C",
        "best_l1_ratio",
        "auroc",
        "f1",
        "sensitivity",
        "specificity",
    ]

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter="\t",
        )

        writer.writeheader()

        for fold in result.folds:
            writer.writerow({
                "fold": (
                    fold.fold
                ),
                "train_n": (
                    fold.train_n
                ),
                "valid_n": (
                    fold.valid_n
                ),
                "best_inner_auroc": (
                    fold.best_inner_auroc
                ),
                "best_C": (
                    fold.best_C
                ),
                "best_l1_ratio": (
                    fold.best_l1_ratio
                ),
                "auroc": (
                    fold.auroc
                ),
                "f1": (
                    fold.f1
                ),
                "sensitivity": (
                    fold.sensitivity
                ),
                "specificity": (
                    fold.specificity
                ),
            })


def _write_summary(
    cv_result: BaselineCVResult,
    final_result: FinalBaselineResult,
    data: CohortData,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "summary.json"
    )

    payload = {
        "cohort": DEVELOPMENT,

        "representation": (
            "all_shared_genes"
        ),

        "gene_count": int(
            data.X.shape[1]
        ),

        "feature_selection": None,

        "scaling": (
            "StandardScaler"
        ),

        "model": (
            "LogisticRegression elastic-net"
        ),

        "solver": "saga",

        "outer_cv": {
            "type": (
                "StratifiedKFold"
            ),
            "n_splits": (
                OUTER_SPLITS
            ),
            "shuffle": True,
            "random_state": (
                OUTER_RANDOM_STATE
            ),
        },

        "inner_cv": {
            "type": (
                "StratifiedKFold"
            ),
            "n_splits": (
                INNER_SPLITS
            ),
            "shuffle": True,
            "random_state": (
                INNER_RANDOM_STATE
            ),
        },

        "selection_metric": (
            "roc_auc"
        ),

        "C_grid": C_GRID,

        "l1_ratio_grid": (
            L1_RATIO_GRID
        ),

        "classification_threshold": (
            CLASSIFICATION_THRESHOLD
        ),

        "class_weight": None,

        "nested_cv_summary": (
            cv_result.summary
        ),

        "final_model": {
            "best_inner_auroc": (
                final_result.best_inner_auroc
            ),
            "best_C": (
                final_result.best_C
            ),
            "best_l1_ratio": (
                final_result.best_l1_ratio
            ),
        },

        "external_data_used": False,
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_baseline_results(
    cv_result: BaselineCVResult,
    final_result: FinalBaselineResult,
    data: CohortData,
    output_dir: Path = (
        DEFAULT_OUTPUT_DIR
    ),
) -> Path:
    """Persist baseline results and fitted model."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_fold_metrics(
        cv_result,
        output_dir,
    )

    joblib.dump(
        final_result.model,
        output_dir
        / "all_genes_model.joblib",
    )

    _write_summary(
        cv_result,
        final_result,
        data,
        output_dir,
    )

    return output_dir


def print_cv_result(
    result: BaselineCVResult,
) -> None:
    print(
        "All-shared-genes nested CV: "
        "GSE98793 only"
    )

    print(
        f"genes=16,467, "
        f"outer={OUTER_SPLITS}, "
        f"inner={INNER_SPLITS}, "
        f"grid="
        f"{len(C_GRID) * len(L1_RATIO_GRID)}"
    )

    for fold in result.folds:
        print(
            f"fold {fold.fold}: "
            f"train={fold.train_n}, "
            f"valid={fold.valid_n}, "
            f"best C="
            f"{fold.best_C:g}, "
            f"best l1_ratio="
            f"{fold.best_l1_ratio:g}, "
            f"inner AUROC="
            f"{fold.best_inner_auroc:.3f}, "
            f"outer AUROC="
            f"{fold.auroc:.3f}, "
            f"F1="
            f"{fold.f1:.3f}, "
            f"sensitivity="
            f"{fold.sensitivity:.3f}, "
            f"specificity="
            f"{fold.specificity:.3f}"
        )

    print(
        "\nOuter-fold summary:"
    )

    for metric, values in (
        result.summary.items()
    ):
        print(
            f"{metric}: "
            f"{values['mean']:.3f} "
            f"± "
            f"{values['std']:.3f}"
        )


def main() -> None:
    data = load_cohort(
        DEVELOPMENT
    )

    cv_result = (
        run_nested_baseline_cv(
            data
        )
    )

    print_cv_result(
        cv_result
    )

    print(
        "\nFitting final "
        "all-shared-genes baseline..."
    )

    final_result = (
        fit_final_baseline(
            data
        )
    )

    print(
        f"final best inner AUROC="
        f"{final_result.best_inner_auroc:.3f}"
    )

    print(
        f"final best C="
        f"{final_result.best_C:g}"
    )

    print(
        f"final best l1_ratio="
        f"{final_result.best_l1_ratio:g}"
    )

    output_dir = (
        save_baseline_results(
            cv_result,
            final_result,
            data,
        )
    )

    print(
        "\nSaved baseline outputs to: "
        f"{output_dir}"
    )

    print(
        "  fold_metrics.tsv"
    )
    print(
        "  summary.json"
    )
    print(
        "  all_genes_model.joblib"
    )


if __name__ == "__main__":
    main()