"""Reference quantile normalization for the predefined sensitivity analysis.

This transformer learns one target quantile distribution from training samples
only. Each sample is then mapped to that fixed target distribution according to
within-sample rank.

Input convention in this project:
- rows = samples
- columns = genes

Because the transformer is used as the first step of an sklearn Pipeline, fit()
is called only on the current training split during cross-validation. transform()
then applies the learned reference unchanged to validation or external samples.

Tied expression values are assigned the mean target value across the tied rank
positions, which avoids arbitrary differences caused only by sort order.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


FloatArray = NDArray[np.float64]


def _as_finite_2d_array(
    X: ArrayLike,
) -> FloatArray:
    """Convert input to a finite 2D float array."""
    array = np.asarray(
        X,
        dtype=float,
    )

    if array.ndim != 2:
        raise ValueError(
            "ReferenceQuantileNormalizer expects a 2D "
            "samples x features matrix."
        )

    if array.shape[0] == 0:
        raise ValueError(
            "ReferenceQuantileNormalizer requires "
            "at least one sample."
        )

    if array.shape[1] == 0:
        raise ValueError(
            "ReferenceQuantileNormalizer requires "
            "at least one feature."
        )

    if not np.all(
        np.isfinite(array)
    ):
        raise ValueError(
            "ReferenceQuantileNormalizer does not support "
            "NaN or infinite values."
        )

    return array


def _normalize_sample(
    sample: FloatArray,
    reference: FloatArray,
) -> FloatArray:
    """Map one sample to the fixed reference distribution."""
    order = np.argsort(
        sample,
        kind="mergesort",
    )

    sorted_sample = sample[
        order
    ]

    normalized_sorted = np.array(
        reference,
        dtype=float,
        copy=True,
    )

    start = 0
    n_features = len(
        sorted_sample
    )

    while start < n_features:
        end = start + 1

        while (
            end < n_features
            and sorted_sample[end]
            == sorted_sample[start]
        ):
            end += 1

        if end - start > 1:
            normalized_sorted[
                start:end
            ] = float(
                np.mean(
                    reference[
                        start:end
                    ]
                )
            )

        start = end

    normalized = np.empty(
        n_features,
        dtype=float,
    )

    normalized[
        order
    ] = normalized_sorted

    return normalized


class ReferenceQuantileNormalizer(
    TransformerMixin,
    BaseEstimator,
):
    """Training-fitted target/reference quantile normalizer."""

    n_features_in_: int
    reference_quantiles_: FloatArray

    def fit(
        self,
        X: ArrayLike,
        y: object | None = None,
    ) -> Self:
        """Learn the reference distribution from training samples."""
        del y

        array = (
            _as_finite_2d_array(
                X
            )
        )

        self.n_features_in_ = int(
            array.shape[1]
        )

        sorted_samples = np.sort(
            array,
            axis=1,
        )

        reference = np.mean(
            sorted_samples,
            axis=0,
            dtype=float,
        )

        if not np.all(
            np.isfinite(
                reference
            )
        ):
            raise ValueError(
                "Learned reference quantiles "
                "contain nonfinite values."
            )

        if np.any(
            np.diff(
                reference
            )
            < 0
        ):
            raise AssertionError(
                "Learned reference distribution "
                "is not monotonic."
            )

        self.reference_quantiles_ = (
            np.asarray(
                reference,
                dtype=float,
            )
        )

        return self

    def transform(
        self,
        X: ArrayLike,
    ) -> FloatArray:
        """Apply the learned training reference to new samples."""
        check_is_fitted(
            self,
            attributes=[
                "n_features_in_",
                "reference_quantiles_",
            ],
        )

        array = (
            _as_finite_2d_array(
                X
            )
        )

        if (
            array.shape[1]
            != self.n_features_in_
        ):
            raise ValueError(
                "Feature count does not match the data "
                "used to fit the normalizer: expected "
                f"{self.n_features_in_}, "
                f"got {array.shape[1]}."
            )

        transformed = (
            np.empty_like(
                array,
                dtype=float,
            )
        )

        for sample_index in range(
            array.shape[0]
        ):
            transformed[
                sample_index
            ] = _normalize_sample(
                array[
                    sample_index
                ],
                self.reference_quantiles_,
            )

        return transformed