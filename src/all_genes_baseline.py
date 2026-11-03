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

Progress reporting:
- prints when each outer fold starts and finishes
- prints a heartbeat every five minutes during long GridSearchCV calls
- reports the final full-development search separately
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
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


DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "all_genes_baseline"
HEARTBEAT_SECONDS = 300


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


def log(message: str) -> None:
    """Print an immediate timestamped progress message."""
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def format_elapsed(seconds: float) -> str:
    """Format elapsed wall-clock time."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"

    if minutes:
        return f"{minutes}m {seconds}s"

    return f"{seconds}s"


def _heartbeat_worker(
    stop_event: threading.Event,
    label: str,
    started: float,
) -> None:
    """Report that a long-running search is still active."""
    while not stop_event.wait(HEARTBEAT_SECONDS):
        elapsed = format_elapsed(
            time.perf_counter() - started
        )

        log(
            f"{label} still running "
            f"— elapsed {elapsed}"
        )


def _fit_search_with_progress(
    search: GridSearchCV,
    X: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
) -> None:
    """Fit GridSearchCV while emitting periodic heartbeat messages."""
    started = time.perf_counter()

    stop_event = threading.Event()

    heartbeat = threading.Thread(
        target=_heartbeat_worker,
        args=(
            stop_event,
            label,
            started,
        ),
        daemon=True,
    )

    log(f"{label} started")

    heartbeat.start()

    try:
        search.fit(
            X,
            y,
        )
    finally:
        stop_event.set()

        heartbeat.join(
            timeout=1.0
        )

    elapsed = format_elapsed(
        time.perf_counter()
        - started
    )

    log(
        f"{label} finished in "
        f"{elapsed}"
    )


def build_baseline_pipeline() -> Pipeline:
    """Build the all-shared-genes pipeline."""
    return build_model_pipeline(
        selector="passthrough"
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
                getattr(
                    fold,
                    metric,
                )
                for fold
                in fold_results
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


def _validate_development_data(
    data: CohortData,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
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
            "X must be a 2D "
            "samples x genes matrix"
        )

    if y.ndim != 1:
        raise ValueError(
            "y must be a 1D "
            "label vector"
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

    if set(
        np.unique(
            y
        )
    ) != {
        0,
        1,
    }:
        raise ValueError(
            "Both binary classes "
            "are required"
        )

    return X, y


def run_nested_baseline_cv(
    data: CohortData,
) -> BaselineCVResult:
    """Estimate baseline performance with nested CV."""
    X, y = (
        _validate_development_data(
            data
        )
    )

    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=(
            OUTER_RANDOM_STATE
        ),
    )

    fold_results: list[
        BaselineFoldResult
    ] = []

    combinations = (
        len(C_GRID)
        * len(
            L1_RATIO_GRID
        )
    )

    fits_per_fold = (
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
        fold_started = (
            time.perf_counter()
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
            f"= {fits_per_fold} fits"
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

        _fit_search_with_progress(
            search,
            X_train,
            y_train,
            label=(
                "Outer fold "
                f"{fold_number}/"
                f"{OUTER_SPLITS} "
                "inner search"
            ),
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

        fold_result = (
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
            )
        )

        fold_results.append(
            fold_result
        )

        elapsed = (
            format_elapsed(
                time.perf_counter()
                - fold_started
            )
        )

        log(
            "Outer fold "
            f"{fold_number}/"
            f"{OUTER_SPLITS} "
            f"complete in {elapsed} "
            "| inner AUROC="
            f"{fold_result.best_inner_auroc:.3f} "
            "| outer AUROC="
            f"{fold_result.auroc:.3f} "
            "| F1="
            f"{fold_result.f1:.3f} "
            "| sensitivity="
            f"{fold_result.sensitivity:.3f} "
            "| specificity="
            f"{fold_result.specificity:.3f} "
            "| C="
            f"{fold_result.best_C:g} "
            "| l1_ratio="
            f"{fold_result.best_l1_ratio:g}"
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
    X, y = (
        _validate_development_data(
            data
        )
    )

    combinations = (
        len(C_GRID)
        * len(
            L1_RATIO_GRID
        )
    )

    total_fits = (
        combinations
        * INNER_SPLITS
    )

    log(
        "Final all-genes search: "
        f"{combinations} combinations "
        f"x {INNER_SPLITS} folds "
        f"= {total_fits} CV fits "
        "+ final refit"
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

    _fit_search_with_progress(
        search,
        X,
        y,
        label=(
            "Final all-genes "
            "development search"
        ),
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
    """Print the completed nested-CV summary."""
    print(
        "All-shared-genes nested CV: "
        "GSE98793 only",
        flush=True,
    )

    print(
        f"genes=16,467, "
        f"outer={OUTER_SPLITS}, "
        f"inner={INNER_SPLITS}, "
        f"grid="
        f"{len(C_GRID) * len(L1_RATIO_GRID)}",
        flush=True,
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
            f"{fold.specificity:.3f}",
            flush=True,
        )

    print(
        "\nOuter-fold summary:",
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
        "Starting all-shared-genes "
        "nested CV"
    )

    cv_result = (
        run_nested_baseline_cv(
            data
        )
    )

    print_cv_result(
        cv_result
    )

    log(
        "Nested CV complete; "
        "starting final "
        "all-shared-genes baseline"
    )

    final_result = (
        fit_final_baseline(
            data
        )
    )

    print(
        f"final best inner AUROC="
        f"{final_result.best_inner_auroc:.3f}",
        flush=True,
    )

    print(
        f"final best C="
        f"{final_result.best_C:g}",
        flush=True,
    )

    print(
        f"final best l1_ratio="
        f"{final_result.best_l1_ratio:g}",
        flush=True,
    )

    output_dir = (
        save_baseline_results(
            cv_result,
            final_result,
            data,
        )
    )

    log(
        "Saved baseline outputs to "
        f"{output_dir}"
    )

    print(
        "  fold_metrics.tsv",
        flush=True,
    )

    print(
        "  summary.json",
        flush=True,
    )

    print(
        "  all_genes_model.joblib",
        flush=True,
    )


if __name__ == "__main__":
    main()