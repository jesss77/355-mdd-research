"""Assemble the shared sklearn model pipeline without fixing open choices.

The approved experiment requires the same model-selection procedure for the
gene-level and pathway-level representations. This module gives both sides the
same named pipeline structure:

    preprocess -> select -> scale -> model

Any unresolved step can remain "passthrough" until the methodology is frozen.
Because sklearn Pipeline fits steps in order, fitted transformations stay inside
the training fold when the pipeline is used by cv_framework.py.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.modeling import build_elastic_net_model


def build_model_pipeline(
    *,
    preprocessor="passthrough",
    selector="passthrough",
    scaler="passthrough",
    C: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    """Create the common pipeline used by gene/pathway representations.

    Open methodology choices are passed in explicitly rather than hardcoded.
    """
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
