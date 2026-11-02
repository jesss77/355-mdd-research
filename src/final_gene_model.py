"""Fit and save the final development-only gene model.

This script uses GSE98793 only.

After nested CV has estimated development performance, the final model is
selected using the same frozen inner 5-fold GridSearchCV on all GSE98793
samples. GridSearchCV then refits the best complete pipeline on all development
samples.

The fitted pipeline contains:
- top-50 ANOVA F gene selector
- StandardScaler
- elastic-net logistic regression

External cohorts are never loaded or evaluated here.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.model_data import DEVELOPMENT, ROOT, CohortData, load_cohort
from src.nested_cv import (
    C_GRID,
    INNER_RANDOM_STATE,
    INNER_SPLITS,
    L1_RATIO_GRID,
    build_gene_pipeline,
    build_parameter_grid,
)


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "gene_final_model"


@dataclass(frozen=True)
class FinalGeneModelResult:
    model: Pipeline
    best_inner_auroc: float
    best_C: float
    best_l1_ratio: float
    selected_gene_ids: tuple[str, ...]
    selected_gene_symbols: tuple[str, ...]
    selected_gene_scores: tuple[float, ...]
    cv_results: tuple[dict[str, float | int], ...]


def fit_final_gene_model(
    data: CohortData,
) -> FinalGeneModelResult:
    """Select and refit the final pipeline using GSE98793 only."""
    if data.cohort != DEVELOPMENT:
        raise ValueError(
            "Final gene-model fitting is restricted to GSE98793."
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
            "X and y must contain the same number of samples"
        )

    if X.shape[1] != len(data.gene_ids):
        raise ValueError(
            "Gene metadata does not match X columns"
        )

    if set(np.unique(y)) != {0, 1}:
        raise ValueError(
            "Both classes 0 and 1 are required"
        )

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
        X,
        y,
    )

    best_model = cast(
        Pipeline,
        search.best_estimator_,
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
            "Expected exactly 50 selected genes, "
            f"got {len(selected_indices)}"
        )

    raw_scores = selector.scores_

    if raw_scores is None:
        raise AssertionError(
            "Final fitted selector has no ANOVA F scores"
        )

    all_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    if all_scores.ndim != 1:
        raise ValueError(
            "ANOVA score array must be one-dimensional"
        )

    if len(all_scores) != X.shape[1]:
        raise ValueError(
            "ANOVA score count does not match gene count"
        )

    selected_scores = all_scores[
        selected_indices
    ]

    if not np.all(
        np.isfinite(selected_scores)
    ):
        raise ValueError(
            "Selected genes contain nonfinite ANOVA scores"
        )

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

    cv_results: list[
        dict[str, float | int]
    ] = []

    means = np.asarray(
        search.cv_results_["mean_test_score"],
        dtype=float,
    )

    stds = np.asarray(
        search.cv_results_["std_test_score"],
        dtype=float,
    )

    ranks = np.asarray(
        search.cv_results_["rank_test_score"],
        dtype=int,
    )

    params = search.cv_results_["params"]

    for index, param_set in enumerate(params):
        cv_results.append({
            "rank": int(
                ranks[index]
            ),
            "C": float(
                param_set["model__C"]
            ),
            "l1_ratio": float(
                param_set["model__l1_ratio"]
            ),
            "mean_inner_auroc": float(
                means[index]
            ),
            "std_inner_auroc": float(
                stds[index]
            ),
        })

    cv_results.sort(
        key=lambda row: int(
            row["rank"]
        )
    )

    return FinalGeneModelResult(
        model=best_model,
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
        selected_gene_ids=(
            selected_gene_ids
        ),
        selected_gene_symbols=(
            selected_gene_symbols
        ),
        selected_gene_scores=(
            selected_gene_scores
        ),
        cv_results=tuple(
            cv_results
        ),
    )


def _write_selected_genes(
    result: FinalGeneModelResult,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "selected_genes.tsv"
    )

    fieldnames = [
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

        for rank, (
            gene_id,
            symbol,
            score,
        ) in enumerate(
            zip(
                result.selected_gene_ids,
                result.selected_gene_symbols,
                result.selected_gene_scores,
                strict=True,
            ),
            start=1,
        ):
            writer.writerow({
                "rank": rank,
                "entrez_gene_id": gene_id,
                "gene_symbol": symbol,
                "anova_f_score": score,
            })


def _write_cv_results(
    result: FinalGeneModelResult,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "grid_search.tsv"
    )

    fieldnames = [
        "rank",
        "C",
        "l1_ratio",
        "mean_inner_auroc",
        "std_inner_auroc",
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

        for row in result.cv_results:
            writer.writerow(row)


def _write_metadata(
    result: FinalGeneModelResult,
    data: CohortData,
    output_dir: Path,
) -> None:
    path = (
        output_dir
        / "metadata.json"
    )

    payload = {
        "cohort": DEVELOPMENT,

        "development_samples": int(
            len(data.y)
        ),

        "class_counts": {
            "MDD": int(
                np.sum(data.y == 1)
            ),
            "control": int(
                np.sum(data.y == 0)
            ),
        },

        "feature_selection": {
            "method": "SelectKBest",
            "score_func": "f_classif",
            "k": 50,
        },

        "scaling": "StandardScaler",

        "model": (
            "LogisticRegression elastic-net"
        ),

        "solver": "saga",

        "inner_cv": {
            "type": "StratifiedKFold",
            "n_splits": INNER_SPLITS,
            "shuffle": True,
            "random_state": (
                INNER_RANDOM_STATE
            ),
        },

        "selection_metric": "roc_auc",

        "C_grid": C_GRID,

        "l1_ratio_grid": (
            L1_RATIO_GRID
        ),

        "best_inner_auroc": (
            result.best_inner_auroc
        ),

        "best_C": (
            result.best_C
        ),

        "best_l1_ratio": (
            result.best_l1_ratio
        ),

        "selected_gene_count": len(
            result.selected_gene_ids
        ),

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


def save_final_gene_model(
    result: FinalGeneModelResult,
    data: CohortData,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Save final fitted pipeline and its audit trail."""
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        result.model,
        output_dir / "gene_model.joblib",
    )

    _write_selected_genes(
        result,
        output_dir,
    )

    _write_cv_results(
        result,
        output_dir,
    )

    _write_metadata(
        result,
        data,
        output_dir,
    )

    return output_dir


def print_final_gene_model(
    result: FinalGeneModelResult,
) -> None:
    """Print the chosen development-only final model."""
    print(
        "Final gene model: GSE98793 only"
    )

    print(
        f"best inner AUROC="
        f"{result.best_inner_auroc:.3f}"
    )

    print(
        f"best C="
        f"{result.best_C:g}"
    )

    print(
        f"best l1_ratio="
        f"{result.best_l1_ratio:g}"
    )

    print(
        f"selected genes="
        f"{len(result.selected_gene_ids)}"
    )

    print(
        "\nTop 10 selected genes:"
    )

    for rank, (
        symbol,
        gene_id,
        score,
    ) in enumerate(
        zip(
            result.selected_gene_symbols,
            result.selected_gene_ids,
            result.selected_gene_scores,
            strict=True,
        ),
        start=1,
    ):
        if rank > 10:
            break

        print(
            f"{rank:2d}. "
            f"{symbol} "
            f"(Entrez {gene_id}) "
            f"F={score:.3f}"
        )


def main() -> None:
    data = load_cohort(
        DEVELOPMENT
    )

    result = fit_final_gene_model(
        data
    )

    print_final_gene_model(
        result
    )

    output_dir = save_final_gene_model(
        result,
        data,
    )

    print(
        "\nSaved final model to: "
        f"{output_dir}"
    )

    print(
        "  gene_model.joblib"
    )
    print(
        "  selected_genes.tsv"
    )
    print(
        "  grid_search.tsv"
    )
    print(
        "  metadata.json"
    )


if __name__ == "__main__":
    main()