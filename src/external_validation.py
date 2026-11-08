"""Frozen external validation for the CMPS 355 MDD experiment.

Primary question:
Do pathway-level transcriptomic representations improve cross-cohort
generalization compared with gene-level representations?

Development cohort:
- GSE98793

External cohorts:
- GSE32280
- GSE39653

Frozen gene model:
- top 50 genes by ANOVA F statistic
- StandardScaler
- elastic-net logistic regression
- C = 0.01
- l1_ratio = 0.1

Frozen pathway model:
- 50 Hallmark ssGSEA raw ES features
- StandardScaler
- elastic-net logistic regression
- C = 10
- l1_ratio = 0.9

No external sample is used for feature selection, fitting, scaling,
hyperparameter tuning, or threshold selection.

External uncertainty:
- 10,000 class-stratified bootstrap replicates
- 95% percentile confidence intervals
- paired pathway-minus-gene AUROC bootstrap
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest

from src.gene_features import build_gene_selector
from src.model_data import (
    DEVELOPMENT,
    EXTERNAL,
    ROOT,
    CohortData,
    load_cohort,
)
from src.modeling import classification_metrics
from src.pipeline_factory import build_model_pipeline


GENE_C = 0.01
GENE_L1_RATIO = 0.1

PATHWAY_C = 10.0
PATHWAY_L1_RATIO = 0.9

CLASSIFICATION_THRESHOLD = 0.5

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_RANDOM_STATE = 44
CI_LEVEL = 0.95

PATHWAY_FEATURE_DIR = (
    ROOT
    / "data"
    / "features"
)

PATHWAY_SUMMARY_PATH = (
    ROOT
    / "results"
    / "pathway"
    / "summary.json"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "external"
)

MODEL_DIR = (
    ROOT
    / "outputs"
    / "external_validation"
)


@dataclass(frozen=True)
class PathwayData:
    cohort: str
    X: np.ndarray
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]


def load_pathway_features(
    cohort: str,
    expected_sample_ids: tuple[str, ...],
    expected_feature_names: tuple[str, ...] | None = None,
) -> PathwayData:
    """Load one committed Hallmark ssGSEA feature matrix."""

    path = (
        PATHWAY_FEATURE_DIR
        / f"{cohort}_hallmark_ssgsea.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing pathway features: {path}"
        )

    with path.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise ValueError(
                f"{cohort}: empty pathway header"
            )

        if reader.fieldnames[0] != "GSM":
            raise ValueError(
                f"{cohort}: first pathway column "
                "must be GSM"
            )

        feature_names = tuple(
            reader.fieldnames[1:]
        )

        if len(feature_names) != 50:
            raise ValueError(
                f"{cohort}: expected 50 pathway "
                f"features, got {len(feature_names)}"
            )

        if len(set(feature_names)) != 50:
            raise ValueError(
                f"{cohort}: duplicate pathway names"
            )

        if (
            expected_feature_names is not None
            and feature_names
            != expected_feature_names
        ):
            raise ValueError(
                f"{cohort}: pathway feature names/"
                "order differ from development"
            )

        rows_by_sample: dict[
            str,
            np.ndarray,
        ] = {}

        for line_no, row in enumerate(
            reader,
            start=2,
        ):
            sample_id = row["GSM"]

            if not sample_id:
                raise ValueError(
                    f"{cohort}: missing GSM "
                    f"on line {line_no}"
                )

            if sample_id in rows_by_sample:
                raise ValueError(
                    f"{cohort}: duplicate GSM "
                    f"{sample_id}"
                )

            try:
                values = np.asarray(
                    [
                        float(row[name])
                        for name
                        in feature_names
                    ],
                    dtype=np.float64,
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{cohort}: nonnumeric pathway "
                    f"value on line {line_no}"
                ) from exc

            if values.shape != (50,):
                raise ValueError(
                    f"{cohort}: unexpected pathway "
                    f"row shape {values.shape}"
                )

            if not np.all(
                np.isfinite(values)
            ):
                raise ValueError(
                    f"{cohort}: nonfinite pathway "
                    f"value on line {line_no}"
                )

            rows_by_sample[
                sample_id
            ] = values

    expected_set = set(
        expected_sample_ids
    )

    observed_set = set(
        rows_by_sample
    )

    missing = sorted(
        expected_set
        - observed_set
    )

    extra = sorted(
        observed_set
        - expected_set
    )

    if missing or extra:
        raise ValueError(
            f"{cohort}: pathway/manifest "
            f"sample mismatch; "
            f"missing={missing}, "
            f"extra={extra}"
        )

    X = np.vstack(
        [
            rows_by_sample[
                sample_id
            ]
            for sample_id
            in expected_sample_ids
        ]
    )

    if X.shape != (
        len(expected_sample_ids),
        50,
    ):
        raise AssertionError(
            f"{cohort}: unexpected pathway "
            f"matrix shape {X.shape}"
        )

    return PathwayData(
        cohort=cohort,
        X=X,
        sample_ids=expected_sample_ids,
        feature_names=feature_names,
    )


def verify_frozen_pathway_model() -> None:
    """Verify the committed pathway result matches the frozen model."""

    if not PATHWAY_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PATHWAY_SUMMARY_PATH}"
        )

    payload = json.loads(
        PATHWAY_SUMMARY_PATH.read_text(
            encoding="utf-8"
        )
    )

    final_model = payload[
        "final_model"
    ]

    observed_C = float(
        final_model["C"]
    )

    observed_l1_ratio = float(
        final_model[
            "l1_ratio"
        ]
    )

    observed_pathway_count = int(
        final_model[
            "nonzero_pathways"
        ]
    )

    if observed_C != PATHWAY_C:
        raise ValueError(
            "Frozen pathway C mismatch: "
            f"expected {PATHWAY_C}, "
            f"found {observed_C}"
        )

    if (
        observed_l1_ratio
        != PATHWAY_L1_RATIO
    ):
        raise ValueError(
            "Frozen pathway l1_ratio mismatch: "
            f"expected {PATHWAY_L1_RATIO}, "
            f"found {observed_l1_ratio}"
        )

    if observed_pathway_count != 50:
        raise ValueError(
            "Frozen pathway count mismatch: "
            f"expected 50, "
            f"found {observed_pathway_count}"
        )


def verify_gene_alignment(
    development: CohortData,
    external: CohortData,
) -> None:
    """Ensure external gene columns match development exactly."""

    if (
        external.gene_ids
        != development.gene_ids
    ):
        raise ValueError(
            f"{external.cohort}: gene IDs/order "
            "do not exactly match GSE98793"
        )

    if (
        external.X.shape[1]
        != development.X.shape[1]
    ):
        raise ValueError(
            f"{external.cohort}: gene count "
            "does not match GSE98793"
        )


def fit_frozen_models(
    development: CohortData,
    development_pathway: PathwayData,
):
    """Fit frozen final configurations on GSE98793."""

    gene_model = (
        build_model_pipeline(
            selector=build_gene_selector(
                k=50
            ),
            C=GENE_C,
            l1_ratio=GENE_L1_RATIO,
        )
    )

    pathway_model = (
        build_model_pipeline(
            selector="passthrough",
            C=PATHWAY_C,
            l1_ratio=PATHWAY_L1_RATIO,
        )
    )

    gene_model.fit(
        development.X,
        development.y,
    )

    pathway_model.fit(
        development_pathway.X,
        development.y,
    )

    selector = (
        gene_model
        .named_steps[
            "select"
        ]
    )

    if not isinstance(
        selector,
        SelectKBest,
    ):
        raise TypeError(
            "Gene pipeline selector is not "
            "SelectKBest"
        )

    selected_indices = (
        np.flatnonzero(
            selector.get_support()
        )
    )

    if len(selected_indices) != 50:
        raise AssertionError(
            "Expected 50 final selected genes, "
            f"got {len(selected_indices)}"
        )

    return (
        gene_model,
        pathway_model,
        selected_indices,
    )


def compute_metrics(
    y: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[
    dict[str, float],
    dict[str, int],
    np.ndarray,
]:
    """Compute proposal metrics plus balanced accuracy."""

    metrics = (
        classification_metrics(
            y,
            probabilities,
            threshold=(
                CLASSIFICATION_THRESHOLD
            ),
        )
    )

    predictions = (
        probabilities
        >= CLASSIFICATION_THRESHOLD
    ).astype(
        np.int8
    )

    tn = int(
        np.sum(
            (y == 0)
            & (
                predictions
                == 0
            )
        )
    )

    fp = int(
        np.sum(
            (y == 0)
            & (
                predictions
                == 1
            )
        )
    )

    fn = int(
        np.sum(
            (y == 1)
            & (
                predictions
                == 0
            )
        )
    )

    tp = int(
        np.sum(
            (y == 1)
            & (
                predictions
                == 1
            )
        )
    )

    metrics = dict(
        metrics
    )

    metrics[
        "balanced_accuracy"
    ] = (
        metrics[
            "sensitivity"
        ]
        + metrics[
            "specificity"
        ]
    ) / 2.0

    confusion = {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    return (
        metrics,
        confusion,
        predictions,
    )


def stratified_bootstrap_indices(
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample controls and MDD samples separately."""

    control_indices = (
        np.flatnonzero(
            y == 0
        )
    )

    mdd_indices = (
        np.flatnonzero(
            y == 1
        )
    )

    if (
        len(control_indices) == 0
        or len(mdd_indices) == 0
    ):
        raise ValueError(
            "Both classes are required "
            "for bootstrap evaluation"
        )

    sampled_controls = (
        rng.choice(
            control_indices,
            size=len(
                control_indices
            ),
            replace=True,
        )
    )

    sampled_mdd = (
        rng.choice(
            mdd_indices,
            size=len(
                mdd_indices
            ),
            replace=True,
        )
    )

    return np.concatenate(
        [
            sampled_controls,
            sampled_mdd,
        ]
    )


def bootstrap_comparison(
    y: np.ndarray,
    gene_probabilities: np.ndarray,
    pathway_probabilities: np.ndarray,
) -> tuple[
    dict[
        str,
        dict[
            str,
            tuple[
                float,
                float,
            ],
        ],
    ],
    tuple[
        float,
        float,
    ],
]:
    """Compute class-stratified paired bootstrap intervals."""

    rng = (
        np.random.default_rng(
            BOOTSTRAP_RANDOM_STATE
        )
    )

    metric_names = (
        "auroc",
        "f1",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
    )

    values = {
        model: {
            metric: np.empty(
                BOOTSTRAP_REPLICATES,
                dtype=np.float64,
            )
            for metric
            in metric_names
        }
        for model
        in (
            "gene",
            "pathway",
        )
    }

    auroc_difference = (
        np.empty(
            BOOTSTRAP_REPLICATES,
            dtype=np.float64,
        )
    )

    progress_every = max(
        1,
        BOOTSTRAP_REPLICATES
        // 10,
    )

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):
        indices = (
            stratified_bootstrap_indices(
                y,
                rng,
            )
        )

        y_boot = y[
            indices
        ]

        # Some resampled prediction sets can
        # contain no predicted positive cases.
        # sklearn then defines F1 as 0 and may
        # emit an UndefinedMetricWarning.
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                category=(
                    UndefinedMetricWarning
                ),
            )

            (
                gene_metrics,
                _,
                _,
            ) = compute_metrics(
                y_boot,
                gene_probabilities[
                    indices
                ],
            )

            (
                pathway_metrics,
                _,
                _,
            ) = compute_metrics(
                y_boot,
                pathway_probabilities[
                    indices
                ],
            )

        for metric in (
            metric_names
        ):
            values[
                "gene"
            ][
                metric
            ][
                replicate
            ] = (
                gene_metrics[
                    metric
                ]
            )

            values[
                "pathway"
            ][
                metric
            ][
                replicate
            ] = (
                pathway_metrics[
                    metric
                ]
            )

        auroc_difference[
            replicate
        ] = (
            pathway_metrics[
                "auroc"
            ]
            - gene_metrics[
                "auroc"
            ]
        )

        completed = (
            replicate
            + 1
        )

        if (
            completed
            % progress_every
            == 0
            or completed
            == BOOTSTRAP_REPLICATES
        ):
            print(
                "    bootstrap: "
                f"{completed:,}"
                "/"
                f"{BOOTSTRAP_REPLICATES:,}"
            )

    alpha = (
        1.0
        - CI_LEVEL
    ) / 2.0

    percentiles = [
        100.0
        * alpha,
        100.0
        * (
            1.0
            - alpha
        ),
    ]

    intervals: dict[
        str,
        dict[
            str,
            tuple[
                float,
                float,
            ],
        ],
    ] = {
        "gene": {},
        "pathway": {},
    }

    for model in (
        "gene",
        "pathway",
    ):
        for metric in (
            metric_names
        ):
            (
                low,
                high,
            ) = np.percentile(
                values[
                    model
                ][
                    metric
                ],
                percentiles,
            )

            intervals[
                model
            ][
                metric
            ] = (
                float(
                    low
                ),
                float(
                    high
                ),
            )

    (
        diff_low,
        diff_high,
    ) = np.percentile(
        auroc_difference,
        percentiles,
    )

    return (
        intervals,
        (
            float(
                diff_low
            ),
            float(
                diff_high
            ),
        ),
    )


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[
        dict[
            str,
            object,
        ]
    ],
) -> None:
    """Write one reproducible result table."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = (
            csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def save_final_gene_set(
    development: CohortData,
    selected_indices: np.ndarray,
) -> None:
    """Save the final 50 genes selected on all GSE98793 samples."""

    rows: list[
        dict[
            str,
            object,
        ]
    ] = [
        {
            "entrez_gene_id": (
                development
                .gene_ids[
                    index
                ]
            ),
            "gene_symbol": (
                development
                .gene_symbols[
                    index
                ]
            ),
        }
        for index
        in selected_indices
    ]

    write_csv(
        RESULTS_DIR
        / "final_gene_set.csv",
        [
            "entrez_gene_id",
            "gene_symbol",
        ],
        rows,
    )


def main() -> None:
    print(
        "Frozen external validation"
    )

    print(
        "=========================="
    )

    verify_frozen_pathway_model()

    print(
        "Frozen pathway configuration "
        "verified"
    )

    print(
        "\nLoading GSE98793 "
        "development data"
    )

    development = (
        load_cohort(
            DEVELOPMENT
        )
    )

    development_pathway = (
        load_pathway_features(
            DEVELOPMENT,
            development.sample_ids,
        )
    )

    print(
        "Development data:"
    )

    print(
        "  gene matrix: "
        f"{development.X.shape}"
    )

    print(
        "  pathway matrix: "
        f"{development_pathway.X.shape}"
    )

    print(
        "  labels: "
        f"{int(np.sum(development.y == 1))} "
        "MDD / "
        f"{int(np.sum(development.y == 0))} "
        "controls"
    )

    print(
        "\nFitting frozen gene and "
        "pathway models on "
        "GSE98793 only"
    )

    (
        gene_model,
        pathway_model,
        selected_gene_indices,
    ) = fit_frozen_models(
        development,
        development_pathway,
    )

    print(
        "Frozen development models fitted"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        gene_model,
        MODEL_DIR
        / "gene_model.joblib",
    )

    joblib.dump(
        pathway_model,
        MODEL_DIR
        / "pathway_model.joblib",
    )

    save_final_gene_set(
        development,
        selected_gene_indices,
    )

    prediction_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    metric_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    confusion_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    paired_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    cohort_summaries: dict[
        str,
        object,
    ] = {}

    summary: dict[
        str,
        object,
    ] = {
        "design": {
            "development": (
                DEVELOPMENT
            ),
            "external_cohorts": (
                list(
                    EXTERNAL
                )
            ),
            "gene_model": {
                "features": (
                    "top_50_anova_f"
                ),
                "C": (
                    GENE_C
                ),
                "l1_ratio": (
                    GENE_L1_RATIO
                ),
            },
            "pathway_model": {
                "features": (
                    "50_Hallmark_"
                    "ssGSEA_raw_ES"
                ),
                "C": (
                    PATHWAY_C
                ),
                "l1_ratio": (
                    PATHWAY_L1_RATIO
                ),
            },
            "threshold": (
                CLASSIFICATION_THRESHOLD
            ),
            "bootstrap_replicates": (
                BOOTSTRAP_REPLICATES
            ),
            "bootstrap_random_state": (
                BOOTSTRAP_RANDOM_STATE
            ),
            "bootstrap_method": (
                "class_stratified_paired"
            ),
            "ci_level": (
                CI_LEVEL
            ),
        },
        "cohorts": (
            cohort_summaries
        ),
    }

    # External cohorts are intentionally
    # loaded only after both final models
    # have already been fitted.
    for cohort in EXTERNAL:
        print(
            f"\nEvaluating {cohort}"
        )

        external_gene = (
            load_cohort(
                cohort
            )
        )

        verify_gene_alignment(
            development,
            external_gene,
        )

        external_pathway = (
            load_pathway_features(
                cohort,
                external_gene.sample_ids,
                expected_feature_names=(
                    development_pathway
                    .feature_names
                ),
            )
        )

        print(
            "  gene alignment: verified"
        )

        print(
            "  pathway alignment: verified"
        )

        print(
            "  samples: "
            f"{len(external_gene.y)} "
            "("
            f"{int(np.sum(external_gene.y == 1))} "
            "MDD / "
            f"{int(np.sum(external_gene.y == 0))} "
            "controls)"
        )

        gene_probabilities = (
            gene_model
            .predict_proba(
                external_gene.X
            )[:, 1]
        )

        pathway_probabilities = (
            pathway_model
            .predict_proba(
                external_pathway.X
            )[:, 1]
        )

        (
            gene_metrics,
            gene_confusion,
            gene_predictions,
        ) = compute_metrics(
            external_gene.y,
            gene_probabilities,
        )

        (
            pathway_metrics,
            pathway_confusion,
            pathway_predictions,
        ) = compute_metrics(
            external_gene.y,
            pathway_probabilities,
        )

        print(
            "  Running "
            f"{BOOTSTRAP_REPLICATES:,} "
            "paired class-stratified "
            "bootstrap replicates"
        )

        (
            intervals,
            difference_ci,
        ) = bootstrap_comparison(
            external_gene.y,
            gene_probabilities,
            pathway_probabilities,
        )

        point_difference = (
            pathway_metrics[
                "auroc"
            ]
            - gene_metrics[
                "auroc"
            ]
        )

        for (
            sample_id,
            label,
            gene_probability,
            gene_prediction,
            pathway_probability,
            pathway_prediction,
        ) in zip(
            external_gene.sample_ids,
            external_gene.y,
            gene_probabilities,
            gene_predictions,
            pathway_probabilities,
            pathway_predictions,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "cohort": (
                        cohort
                    ),
                    "GSM": (
                        sample_id
                    ),
                    "label": (
                        int(
                            label
                        )
                    ),
                    "gene_probability": (
                        float(
                            gene_probability
                        )
                    ),
                    "gene_prediction": (
                        int(
                            gene_prediction
                        )
                    ),
                    "pathway_probability": (
                        float(
                            pathway_probability
                        )
                    ),
                    "pathway_prediction": (
                        int(
                            pathway_prediction
                        )
                    ),
                }
            )

        for (
            model_name,
            metrics,
        ) in (
            (
                "gene",
                gene_metrics,
            ),
            (
                "pathway",
                pathway_metrics,
            ),
        ):
            for (
                metric,
                estimate,
            ) in metrics.items():
                (
                    low,
                    high,
                ) = intervals[
                    model_name
                ][
                    metric
                ]

                metric_rows.append(
                    {
                        "cohort": (
                            cohort
                        ),
                        "model": (
                            model_name
                        ),
                        "metric": (
                            metric
                        ),
                        "estimate": (
                            float(
                                estimate
                            )
                        ),
                        "ci_low": (
                            low
                        ),
                        "ci_high": (
                            high
                        ),
                    }
                )

        confusion_rows.extend(
            [
                {
                    "cohort": (
                        cohort
                    ),
                    "model": (
                        "gene"
                    ),
                    **gene_confusion,
                },
                {
                    "cohort": (
                        cohort
                    ),
                    "model": (
                        "pathway"
                    ),
                    **pathway_confusion,
                },
            ]
        )

        paired_rows.append(
            {
                "cohort": (
                    cohort
                ),
                "gene_auroc": (
                    gene_metrics[
                        "auroc"
                    ]
                ),
                "pathway_auroc": (
                    pathway_metrics[
                        "auroc"
                    ]
                ),
                "pathway_minus_gene": (
                    point_difference
                ),
                "ci_low": (
                    difference_ci[
                        0
                    ]
                ),
                "ci_high": (
                    difference_ci[
                        1
                    ]
                ),
            }
        )

        cohort_summaries[
            cohort
        ] = {
            "n": int(
                len(
                    external_gene.y
                )
            ),
            "MDD": int(
                np.sum(
                    external_gene.y
                    == 1
                )
            ),
            "control": int(
                np.sum(
                    external_gene.y
                    == 0
                )
            ),
            "gene": {
                "metrics": (
                    gene_metrics
                ),
                "confusion": (
                    gene_confusion
                ),
                "ci": {
                    metric: {
                        "low": (
                            interval[
                                0
                            ]
                        ),
                        "high": (
                            interval[
                                1
                            ]
                        ),
                    }
                    for (
                        metric,
                        interval,
                    ) in intervals[
                        "gene"
                    ].items()
                },
            },
            "pathway": {
                "metrics": (
                    pathway_metrics
                ),
                "confusion": (
                    pathway_confusion
                ),
                "ci": {
                    metric: {
                        "low": (
                            interval[
                                0
                            ]
                        ),
                        "high": (
                            interval[
                                1
                            ]
                        ),
                    }
                    for (
                        metric,
                        interval,
                    ) in intervals[
                        "pathway"
                    ].items()
                },
            },
            "paired_auroc_difference": {
                "pathway_minus_gene": (
                    point_difference
                ),
                "ci_low": (
                    difference_ci[
                        0
                    ]
                ),
                "ci_high": (
                    difference_ci[
                        1
                    ]
                ),
            },
        }

        print(
            "  Gene:    "
            f"AUROC="
            f"{gene_metrics['auroc']:.3f}, "
            f"F1="
            f"{gene_metrics['f1']:.3f}, "
            "sensitivity="
            f"{gene_metrics['sensitivity']:.3f}, "
            "specificity="
            f"{gene_metrics['specificity']:.3f}, "
            "balanced accuracy="
            f"{gene_metrics['balanced_accuracy']:.3f}"
        )

        print(
            "  Pathway: "
            f"AUROC="
            f"{pathway_metrics['auroc']:.3f}, "
            f"F1="
            f"{pathway_metrics['f1']:.3f}, "
            "sensitivity="
            f"{pathway_metrics['sensitivity']:.3f}, "
            "specificity="
            f"{pathway_metrics['specificity']:.3f}, "
            "balanced accuracy="
            f"{pathway_metrics['balanced_accuracy']:.3f}"
        )

        print(
            "  Pathway - gene AUROC: "
            f"{point_difference:+.3f} "
            "(95% bootstrap CI "
            f"{difference_ci[0]:+.3f} "
            "to "
            f"{difference_ci[1]:+.3f})"
        )

    write_csv(
        RESULTS_DIR
        / "predictions.csv",
        [
            "cohort",
            "GSM",
            "label",
            "gene_probability",
            "gene_prediction",
            "pathway_probability",
            "pathway_prediction",
        ],
        prediction_rows,
    )

    write_csv(
        RESULTS_DIR
        / "metrics.csv",
        [
            "cohort",
            "model",
            "metric",
            "estimate",
            "ci_low",
            "ci_high",
        ],
        metric_rows,
    )

    write_csv(
        RESULTS_DIR
        / "confusion_matrices.csv",
        [
            "cohort",
            "model",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
        confusion_rows,
    )

    write_csv(
        RESULTS_DIR
        / "paired_auroc_comparison.csv",
        [
            "cohort",
            "gene_auroc",
            "pathway_auroc",
            "pathway_minus_gene",
            "ci_low",
            "ci_high",
        ],
        paired_rows,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        RESULTS_DIR
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\nSaved external results to:"
    )

    print(
        f"  {RESULTS_DIR}"
    )

    print(
        "  predictions.csv"
    )

    print(
        "  metrics.csv"
    )

    print(
        "  confusion_matrices.csv"
    )

    print(
        "  paired_auroc_comparison.csv"
    )

    print(
        "  final_gene_set.csv"
    )

    print(
        "  summary.json"
    )

    print(
        "\nLocal fitted models saved to:"
    )

    print(
        f"  {MODEL_DIR}"
    )


if __name__ == "__main__":
    main()