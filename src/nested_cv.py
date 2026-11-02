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

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.gene_features import build_gene_selector
from src.model_data import DEVELOPMENT, ROOT, CohortData, load_cohort
from src.modeling import classification_metrics
from src.pipeline_factory import build_model_pipeline


OUTER_SPLITS = 5
INNER_SPLITS = 5
OUTER_RANDOM_STATE = 42
INNER_RANDOM_STATE = 43

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
L1_RATIO_GRID = [0.1, 0.5, 0.9]
CLASSIFICATION_THRESHOLD = 0.5

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gene_nested_cv"


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
    selected_gene_scores: tuple[float, ...]


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
        raise ValueError(
            "X must be a 2D samples x genes matrix"
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a 1D label vector"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of samples"
        )

    if X.shape[1] != len(data.gene_ids):
        raise ValueError(
            "Gene metadata does not match X columns"
        )

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Both binary classes 0 and 1 are required"
        )

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

        search.fit(
            X_train,
            y_train,
        )

        best_model = search.best_estimator_

        probabilities = best_model.predict_proba(
            X_valid
        )[:, 1]

        metrics = classification_metrics(
            y_valid,
            probabilities,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        selector = cast(
            SelectKBest,
            best_model.named_steps["select"],
        )

        selected_indices = np.flatnonzero(
            selector.get_support()
        )

        if len(selected_indices) != 50:
            raise AssertionError(
                f"Outer fold {fold_number}: "
                f"expected 50 selected genes, "
                f"got {len(selected_indices)}"
            )

        raw_scores = selector.scores_

        if raw_scores is None:
            raise AssertionError(
                f"Outer fold {fold_number}: "
                "selector has no fitted ANOVA scores"
            )

        # Convert sklearn's broadly typed scores_ object
        # into a concrete NumPy array before indexed access.
        all_scores = np.asarray(
            raw_scores,
            dtype=float,
        )

        if all_scores.ndim != 1:
            raise ValueError(
                f"Outer fold {fold_number}: "
                "ANOVA score array is not one-dimensional"
            )

        if len(all_scores) != X.shape[1]:
            raise ValueError(
                f"Outer fold {fold_number}: "
                "ANOVA score count does not match gene count"
            )

        selected_scores = all_scores[
            selected_indices
        ]

        if not np.all(
            np.isfinite(selected_scores)
        ):
            raise ValueError(
                f"Outer fold {fold_number}: "
                "selected genes contain nonfinite "
                "ANOVA F scores"
            )

        # Save selected genes from highest to lowest
        # training-fold ANOVA F score.
        score_order = np.argsort(
            selected_scores
        )[::-1]

        ranked_indices = selected_indices[
            score_order
        ]

        ranked_scores = selected_scores[
            score_order
        ]

        selected_gene_ids = tuple(
            data.gene_ids[index]
            for index in ranked_indices
        )

        selected_gene_symbols = tuple(
            data.gene_symbols[index]
            for index in ranked_indices
        )

        selected_gene_scores = tuple(
            float(score)
            for score in ranked_scores
        )

        fold_results.append(
            NestedFoldResult(
                fold=fold_number,
                train_n=len(train_idx),
                valid_n=len(valid_idx),
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
                auroc=metrics["auroc"],
                f1=metrics["f1"],
                sensitivity=metrics[
                    "sensitivity"
                ],
                specificity=metrics[
                    "specificity"
                ],
                selected_gene_ids=(
                    selected_gene_ids
                ),
                selected_gene_symbols=(
                    selected_gene_symbols
                ),
                selected_gene_scores=(
                    selected_gene_scores
                ),
            )
        )

    return NestedCVResult(
        folds=tuple(fold_results),
        summary=_summarize(
            fold_results
        ),
    )


def _write_fold_metrics(
    result: NestedCVResult,
    output_dir: Path,
) -> None:
    path = output_dir / "fold_metrics.tsv"

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
                "fold": fold.fold,
                "train_n": fold.train_n,
                "valid_n": fold.valid_n,
                "best_inner_auroc": (
                    fold.best_inner_auroc
                ),
                "best_C": fold.best_C,
                "best_l1_ratio": (
                    fold.best_l1_ratio
                ),
                "auroc": fold.auroc,
                "f1": fold.f1,
                "sensitivity": (
                    fold.sensitivity
                ),
                "specificity": (
                    fold.specificity
                ),
            })


def _write_selected_genes(
    result: NestedCVResult,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "selected_genes.tsv"
    )

    fieldnames = [
        "fold",
        "rank",
        "entrez_gene_id",
        "gene_symbol",
        "anova_f_score",
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
            for rank, (
                gene_id,
                symbol,
                score,
            ) in enumerate(
                zip(
                    fold.selected_gene_ids,
                    fold.selected_gene_symbols,
                    fold.selected_gene_scores,
                    strict=True,
                ),
                start=1,
            ):
                writer.writerow({
                    "fold": fold.fold,
                    "rank": rank,
                    "entrez_gene_id": (
                        gene_id
                    ),
                    "gene_symbol": symbol,
                    "anova_f_score": score,
                })


def _write_gene_frequency(
    result: NestedCVResult,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "gene_frequency.tsv"
    )

    counts: dict[str, int] = (
        defaultdict(int)
    )

    score_lists: dict[
        str,
        list[float],
    ] = defaultdict(list)

    symbols: dict[str, str] = {}

    for fold in result.folds:
        for (
            gene_id,
            symbol,
            score,
        ) in zip(
            fold.selected_gene_ids,
            fold.selected_gene_symbols,
            fold.selected_gene_scores,
            strict=True,
        ):
            counts[gene_id] += 1

            score_lists[
                gene_id
            ].append(score)

            symbols[
                gene_id
            ] = symbol

    rows = []

    for (
        gene_id,
        fold_count,
    ) in counts.items():
        scores = np.asarray(
            score_lists[gene_id],
            dtype=float,
        )

        rows.append({
            "entrez_gene_id": gene_id,
            "gene_symbol": (
                symbols[gene_id]
            ),
            "fold_count": fold_count,
            "fold_fraction": (
                fold_count
                / OUTER_SPLITS
            ),
            "mean_anova_f_score": (
                float(
                    np.mean(scores)
                )
            ),
        })

    rows.sort(
        key=lambda row: (
            -int(
                row[
                    "fold_count"
                ]
            ),
            -float(
                row[
                    "mean_anova_f_score"
                ]
            ),
            str(
                row[
                    "entrez_gene_id"
                ]
            ),
        )
    )

    fieldnames = [
        "entrez_gene_id",
        "gene_symbol",
        "fold_count",
        "fold_fraction",
        "mean_anova_f_score",
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
        writer.writerows(rows)


def _write_summary_json(
    result: NestedCVResult,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "summary.json"
    )

    payload = {
        "cohort": DEVELOPMENT,

        "design": {
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

            "feature_selection": {
                "method": (
                    "SelectKBest"
                ),
                "score_func": (
                    "f_classif"
                ),
                "k": 50,
            },

            "scaling": (
                "StandardScaler"
            ),

            "model": (
                "LogisticRegression "
                "elastic-net"
            ),

            "solver": "saga",

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
        },

        "summary": result.summary,

        "folds": [
            {
                "fold": fold.fold,
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
                "f1": fold.f1,
                "sensitivity": (
                    fold.sensitivity
                ),
                "specificity": (
                    fold.specificity
                ),
            }
            for fold in result.folds
        ],
    }

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def save_nested_cv_result(
    result: NestedCVResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Persist the nested-CV audit trail."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _write_fold_metrics(
        result,
        output_dir,
    )

    _write_selected_genes(
        result,
        output_dir,
    )

    _write_gene_frequency(
        result,
        output_dir,
    )

    _write_summary_json(
        result,
        output_dir,
    )

    return output_dir


def print_nested_cv_result(
    result: NestedCVResult,
) -> None:
    """Print a compact reproducible summary."""
    print(
        "Nested gene-level CV: "
        "GSE98793 only"
    )

    print(
        f"outer={OUTER_SPLITS} folds, "
        f"inner={INNER_SPLITS} folds, "
        f"grid="
        f"{len(C_GRID) * len(L1_RATIO_GRID)} "
        "combinations"
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
            f"F1={fold.f1:.3f}, "
            f"sensitivity="
            f"{fold.sensitivity:.3f}, "
            f"specificity="
            f"{fold.specificity:.3f}"
        )

    print(
        "\nOuter-fold summary:"
    )

    for (
        metric,
        values,
    ) in result.summary.items():
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

    result = (
        run_nested_gene_cv(
            data
        )
    )

    print_nested_cv_result(
        result
    )

    output_dir = (
        save_nested_cv_result(
            result
        )
    )

    print(
        "\nSaved outputs to: "
        f"{output_dir}"
    )

    print(
        "  fold_metrics.tsv"
    )
    print(
        "  selected_genes.tsv"
    )
    print(
        "  gene_frequency.tsv"
    )
    print(
        "  summary.json"
    )


if __name__ == "__main__":
    main()