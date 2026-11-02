"""Gene-level feature selection for the CMPS 355 MDD experiment.

The approved gene-level representation contains the top 50 genes.

Genes are ranked using the ANOVA F statistic computed only from the
training data supplied to fit(). Because this selector is placed inside
the sklearn Pipeline, feature ranking is repeated independently inside
each cross-validation training fold.
"""

from __future__ import annotations

from sklearn.feature_selection import SelectKBest, f_classif


def build_gene_selector(k: int = 50) -> SelectKBest:
    """Return the training-fitted top-k gene selector."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    return SelectKBest(
        score_func=f_classif,
        k=k,
    )