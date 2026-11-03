"""Reference-QN sensitivity analysis for the gene-level MDD model.

Frozen change relative to the primary gene experiment:

ReferenceQuantileNormalizer
-> top-50 ANOVA genes
-> StandardScaler
-> elastic-net logistic regression

All preprocessing and feature selection are fitted inside training folds only.
External cohorts are never loaded or used here.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
from joblib import Memory
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.final_gene_model import (
    FinalGeneModelResult,
    save_final_gene_model,
)
from src.gene_features import build_gene_selector
from src.model_data import (
    DEVELOPMENT,
    ROOT,
    CohortData,
    load_cohort,
)
from src.modeling import classification_metrics
from src.nested_cv import (
    C_GRID,
    CLASSIFICATION_THRESHOLD,
    INNER_RANDOM_STATE,
    INNER_SPLITS,
    L1_RATIO_GRID,
    NestedCVResult,
    NestedFoldResult,
    OUTER_RANDOM_STATE,
    OUTER_SPLITS,
    build_parameter_grid,
    save_nested_cv_result,
)
from src.pipeline_factory import build_model_pipeline
from src.reference_quantile import ReferenceQuantileNormalizer


OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "gene_sensitivity_qn"
)

NESTED_DIR = (
    OUTPUT_DIR
    / "nested_cv"
)

FINAL_DIR = (
    OUTPUT_DIR
    / "final_model"
)

CACHE_DIR = (
    OUTPUT_DIR
    / "_cache"
)

# There are five inner folds.
# Using five workers lets one full set of folds run in parallel,
# while the cached preprocessing can be reused across parameter sets.
SEARCH_JOBS = INNER_SPLITS
SEARCH_PRE_DISPATCH = INNER_SPLITS


def log(
    message: str,
) -> None:
    """Print an immediate timestamped progress message."""
    print(
        f"[{datetime.now():%H:%M:%S}] "
        f"{message}",
        flush=True,
    )


def format_elapsed(
    seconds: float,
) -> str:
    """Format elapsed wall-clock time."""
    total = int(
        round(seconds)
    )

    hours, remainder = divmod(
        total,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    if hours:
        return (
            f"{hours}h "
            f"{minutes}m "
            f"{seconds}s"
        )

    if minutes:
        return (
            f"{minutes}m "
            f"{seconds}s"
        )

    return f"{seconds}s"


def validate_data(
    data: CohortData,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Validate development-only input."""
    if data.cohort != DEVELOPMENT:
        raise ValueError(
            "Sensitivity analysis is "
            "restricted to GSE98793."
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
            "X must be a 2D "
            "samples x genes matrix."
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a 1D label vector."
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y sample counts "
            "do not match."
        )

    if X.shape[1] != len(
        data.gene_ids
    ):
        raise ValueError(
            "Gene metadata does not "
            "match X columns."
        )

    if set(
        np.unique(y)
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Both classes 0 and 1 "
            "are required."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Expression matrix contains "
            "nonfinite values."
        )

    return X, y


def build_sensitivity_pipeline(
    cache_dir: Path | None = None,
) -> Pipeline:
    """Build QN -> top50 -> scaling -> elastic-net."""
    base = build_model_pipeline(
        preprocessor=(
            ReferenceQuantileNormalizer()
        ),
        selector=(
            build_gene_selector(
                k=50
            )
        ),
    )

    memory = None

    if cache_dir is not None:
        cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        memory = Memory(
            location=str(
                cache_dir
            ),
            verbose=0,
        )

    return Pipeline(
        steps=base.steps,
        memory=memory,
    )


def selected_genes(
    model: Pipeline,
    data: CohortData,
    context: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
]:
    """Extract ranked top-50 genes from a fitted pipeline."""
    selector = cast(
        SelectKBest,
        model.named_steps[
            "select"
        ],
    )

    indices = np.flatnonzero(
        selector.get_support()
    )

    if len(indices) != 50:
        raise AssertionError(
            f"{context}: expected "
            "50 selected genes, "
            f"got {len(indices)}."
        )

    raw_scores = (
        selector.scores_
    )

    if raw_scores is None:
        raise AssertionError(
            f"{context}: selector "
            "has no fitted ANOVA scores."
        )

    all_scores = np.asarray(
        raw_scores,
        dtype=float,
    )

    if (
        all_scores.ndim != 1
        or len(all_scores)
        != len(
            data.gene_ids
        )
    ):
        raise ValueError(
            f"{context}: invalid "
            "ANOVA score array."
        )

    scores = all_scores[
        indices
    ]

    if not np.all(
        np.isfinite(
            scores
        )
    ):
        raise ValueError(
            f"{context}: selected genes "
            "contain nonfinite F scores."
        )

    order = np.argsort(
        scores
    )[::-1]

    indices = indices[
        order
    ]

    scores = scores[
        order
    ]

    return (
        tuple(
            data.gene_ids[
                index
            ]
            for index
            in indices
        ),
        tuple(
            data.gene_symbols[
                index
            ]
            for index
            in indices
        ),
        tuple(
            float(score)
            for score
            in scores
        ),
    )


def summarize(
    folds: list[
        NestedFoldResult
    ],
) -> dict[
    str,
    dict[str, float],
]:
    """Summarize outer-fold metrics."""
    summary: dict[
        str,
        dict[str, float],
    ] = {}

    for metric in (
        "auroc",
        "f1",
        "sensitivity",
        "specificity",
    ):
        values = np.asarray(
            [
                getattr(
                    fold,
                    metric,
                )
                for fold
                in folds
            ],
            dtype=float,
        )

        summary[
            metric
        ] = {
            "mean": float(
                np.mean(
                    values
                )
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        }

    return summary


def run_nested_sensitivity(
    data: CohortData,
) -> NestedCVResult:
    """Run nested development CV with reference QN."""
    X, y = validate_data(
        data
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=(
            OUTER_RANDOM_STATE
        ),
    )

    folds: list[
        NestedFoldResult
    ] = []

    combinations = (
        len(C_GRID)
        * len(
            L1_RATIO_GRID
        )
    )

    fits = (
        combinations
        * INNER_SPLITS
    )

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
        started = (
            time.perf_counter()
        )

        fold_cache = (
            CACHE_DIR
            / (
                "outer_fold_"
                f"{fold_number}"
            )
        )

        log(
            "Outer fold "
            f"{fold_number}/"
            f"{OUTER_SPLITS} "
            "started "
            f"(train={len(train_idx)}, "
            f"valid={len(valid_idx)})"
        )

        log(
            "Outer fold "
            f"{fold_number}/"
            f"{OUTER_SPLITS}: "
            f"{combinations} combinations "
            f"x {INNER_SPLITS} folds "
            f"= {fits} fits"
        )

        inner_cv = (
            StratifiedKFold(
                n_splits=(
                    INNER_SPLITS
                ),
                shuffle=True,
                random_state=(
                    INNER_RANDOM_STATE
                ),
            )
        )

        search = GridSearchCV(
            estimator=(
                build_sensitivity_pipeline(
                    fold_cache
                )
            ),
            param_grid=(
                build_parameter_grid()
            ),
            scoring="roc_auc",
            cv=inner_cv,
            refit=True,
            n_jobs=SEARCH_JOBS,
            pre_dispatch=(
                SEARCH_PRE_DISPATCH
            ),
            return_train_score=False,
            error_score="raise",
        )

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

        search.fit(
            X_train,
            y_train,
        )

        model = cast(
            Pipeline,
            search.best_estimator_,
        )

        probabilities = (
            model.predict_proba(
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

        (
            gene_ids,
            gene_symbols,
            gene_scores,
        ) = selected_genes(
            model,
            data,
            (
                "Outer fold "
                f"{fold_number}"
            ),
        )

        fold = NestedFoldResult(
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
                metrics[
                    "auroc"
                ]
            ),
            f1=(
                metrics[
                    "f1"
                ]
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
            selected_gene_ids=(
                gene_ids
            ),
            selected_gene_symbols=(
                gene_symbols
            ),
            selected_gene_scores=(
                gene_scores
            ),
        )

        folds.append(
            fold
        )

        shutil.rmtree(
            fold_cache,
            ignore_errors=True,
        )

        log(
            "Outer fold "
            f"{fold_number}/"
            f"{OUTER_SPLITS} "
            "complete in "
            f"{format_elapsed(time.perf_counter() - started)} "
            "| inner AUROC="
            f"{fold.best_inner_auroc:.3f} "
            "| outer AUROC="
            f"{fold.auroc:.3f} "
            "| C="
            f"{fold.best_C:g} "
            "| l1_ratio="
            f"{fold.best_l1_ratio:g}"
        )

    return NestedCVResult(
        folds=tuple(
            folds
        ),
        summary=summarize(
            folds
        ),
    )


def fit_final_sensitivity(
    data: CohortData,
) -> FinalGeneModelResult:
    """Fit final QN sensitivity model on all GSE98793."""
    X, y = validate_data(
        data
    )

    combinations = (
        len(C_GRID)
        * len(
            L1_RATIO_GRID
        )
    )

    fits = (
        combinations
        * INNER_SPLITS
    )

    final_cache = (
        CACHE_DIR
        / "final"
    )

    log(
        "Final GSE98793 search started: "
        f"{combinations} combinations "
        f"x {INNER_SPLITS} folds "
        f"= {fits} fits"
    )

    started = (
        time.perf_counter()
    )

    inner_cv = (
        StratifiedKFold(
            n_splits=(
                INNER_SPLITS
            ),
            shuffle=True,
            random_state=(
                INNER_RANDOM_STATE
            ),
        )
    )

    search = GridSearchCV(
        estimator=(
            build_sensitivity_pipeline(
                final_cache
            )
        ),
        param_grid=(
            build_parameter_grid()
        ),
        scoring="roc_auc",
        cv=inner_cv,
        refit=True,
        n_jobs=SEARCH_JOBS,
        pre_dispatch=(
            SEARCH_PRE_DISPATCH
        ),
        return_train_score=False,
        error_score="raise",
    )

    search.fit(
        X,
        y,
    )

    model = cast(
        Pipeline,
        search.best_estimator_,
    )

    (
        gene_ids,
        gene_symbols,
        gene_scores,
    ) = selected_genes(
        model,
        data,
        (
            "Final sensitivity "
            "model"
        ),
    )

    means = np.asarray(
        search.cv_results_[
            "mean_test_score"
        ],
        dtype=float,
    )

    stds = np.asarray(
        search.cv_results_[
            "std_test_score"
        ],
        dtype=float,
    )

    ranks = np.asarray(
        search.cv_results_[
            "rank_test_score"
        ],
        dtype=int,
    )

    rows: list[
        dict[
            str,
            float | int,
        ]
    ] = []

    for index, params in enumerate(
        search.cv_results_[
            "params"
        ]
    ):
        rows.append({
            "rank": int(
                ranks[
                    index
                ]
            ),
            "C": float(
                params[
                    "model__C"
                ]
            ),
            "l1_ratio": float(
                params[
                    "model__l1_ratio"
                ]
            ),
            "mean_inner_auroc": float(
                means[
                    index
                ]
            ),
            "std_inner_auroc": float(
                stds[
                    index
                ]
            ),
        })

    rows.sort(
        key=lambda row: int(
            row[
                "rank"
            ]
        )
    )

    result = (
        FinalGeneModelResult(
            model=model,
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
                gene_ids
            ),
            selected_gene_symbols=(
                gene_symbols
            ),
            selected_gene_scores=(
                gene_scores
            ),
            cv_results=tuple(
                rows
            ),
        )
    )

    # The cache is only a computational optimization.
    # Disable it before the fitted model is persisted.
    result.model.set_params(
        memory=None
    )

    shutil.rmtree(
        final_cache,
        ignore_errors=True,
    )

    log(
        "Final GSE98793 search "
        "complete in "
        f"{format_elapsed(time.perf_counter() - started)} "
        "| inner AUROC="
        f"{result.best_inner_auroc:.3f} "
        "| C="
        f"{result.best_C:g} "
        "| l1_ratio="
        f"{result.best_l1_ratio:g}"
    )

    return result


def add_sensitivity_metadata(
    nested_dir: Path,
    final_dir: Path,
) -> None:
    """Add the sensitivity-specific preprocessing provenance."""
    nested_path = (
        nested_dir
        / "summary.json"
    )

    nested = json.loads(
        nested_path.read_text(
            encoding="utf-8"
        )
    )

    nested[
        "analysis"
    ] = (
        "gene_reference_"
        "quantile_sensitivity"
    )

    nested[
        "design"
    ][
        "preprocessing"
    ] = {
        "method": (
            "ReferenceQuantileNormalizer"
        ),
        "fit_data": (
            "training_only"
        ),
        "external_refit": False,
        "tie_handling": (
            "mean target value "
            "across tied rank positions"
        ),
    }

    nested_path.write_text(
        json.dumps(
            nested,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    final_path = (
        final_dir
        / "metadata.json"
    )

    final = json.loads(
        final_path.read_text(
            encoding="utf-8"
        )
    )

    final[
        "analysis"
    ] = (
        "gene_reference_"
        "quantile_sensitivity"
    )

    final[
        "preprocessing"
    ] = {
        "method": (
            "ReferenceQuantileNormalizer"
        ),
        "fit_data": (
            "all_GSE98793_"
            "for_final_model"
        ),
        "external_refit": False,
    }

    final_path.write_text(
        json.dumps(
            final,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def print_summary(
    result: NestedCVResult,
) -> None:
    """Print final outer-CV summary."""
    print(
        "\nReference-QN gene "
        "sensitivity nested CV:",
        flush=True,
    )

    for metric, values in (
        result.summary.items()
    ):
        print(
            f"{metric}: "
            f"{values['mean']:.3f} "
            f"± "
            f"{values['std']:.3f}",
            flush=True,
        )


def main() -> None:
    log(
        "Loading GSE98793 "
        "development data"
    )

    data = load_cohort(
        DEVELOPMENT
    )

    log(
        "Starting reference-QN "
        "gene sensitivity nested CV"
    )

    nested_result = (
        run_nested_sensitivity(
            data
        )
    )

    print_summary(
        nested_result
    )

    nested_dir = (
        save_nested_cv_result(
            nested_result,
            output_dir=(
                NESTED_DIR
            ),
        )
    )

    log(
        "Nested-CV outputs "
        f"saved to {nested_dir}"
    )

    final_result = (
        fit_final_sensitivity(
            data
        )
    )

    final_dir = (
        save_final_gene_model(
            final_result,
            data,
            output_dir=(
                FINAL_DIR
            ),
        )
    )

    add_sensitivity_metadata(
        nested_dir,
        final_dir,
    )

    shutil.rmtree(
        CACHE_DIR,
        ignore_errors=True,
    )

    log(
        "Final sensitivity model "
        f"saved to {final_dir}"
    )

    log(
        "Reference-QN gene "
        "sensitivity analysis complete"
    )


if __name__ == "__main__":
    main()