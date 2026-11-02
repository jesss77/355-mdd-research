"""Assemble the shared sklearn model pipeline.

Primary preprocessing decision:
- retain each cohort's deposited processed expression values
- learn scaling from development/training data only
- apply the learned transformation unchanged to validation/external data

The pipeline uses StandardScaler for per-feature z-scoring. When this pipeline
is passed to the CV framework, sklearn fits the scaler separately inside each
training fold, preventing validation leakage.

Still configurable:
- any additional preprocessor (primary analysis: passthrough)
- feature selector / gene-ranking method
- elastic-net hyperparameters
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import build_elastic_net_model


def build_model_pipeline(
    *,
    preprocessor="passthrough",
    selector="passthrough",
    scaler=None,
    C: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    """Create the common gene/pathway modeling pipeline."""
    if scaler is None:
        scaler = StandardScaler()

    model = build_elastic_net_model(
        C=C,
        l1_ratio=l1_ratio,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("select", selector),
            ("scale", scaler),
            ("model", model),
        ]
    )
