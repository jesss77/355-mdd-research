"""Post-review training-variability analysis."""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.external_validation import load_pathway_features, verify_gene_alignment
from src.gene_features import build_gene_selector
from src.model_data import DEVELOPMENT, EXTERNAL, ROOT, load_cohort
from src.modeling import classification_metrics
from src.nested_cv import (
    INNER_RANDOM_STATE,
    INNER_SPLITS,
    OUTER_RANDOM_STATE,
    OUTER_SPLITS,
    build_parameter_grid,
)
from src.pipeline_factory import build_model_pipeline

RESULTS_DIR = ROOT / "results" / "reviewer_training_variability"


def make_search(gene: bool) -> GridSearchCV:
    selector = build_gene_selector(k=50) if gene else "passthrough"
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=INNER_RANDOM_STATE,
    )
    return GridSearchCV(
        build_model_pipeline(selector=selector),
        build_parameter_grid(),
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )


def metrics_row(fold, model, evaluation, train_n, valid_n, search, y, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        m = classification_metrics(y, p)
    return {
        "fold": fold,
        "model": model,
        "evaluation": evaluation,
        "train_n": train_n,
        "valid_n": valid_n,
        "best_inner_auroc": float(search.best_score_),
        "C": float(search.best_params_["model__C"]),
        "l1_ratio": float(search.best_params_["model__l1_ratio"]),
        **m,
    }


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)),
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dev = load_cohort(DEVELOPMENT)
    dev_path = load_pathway_features(DEVELOPMENT, dev.sample_ids)

    ext_gene = {}
    ext_path = {}
    for cohort in EXTERNAL:
        g = load_cohort(cohort)
        verify_gene_alignment(dev, g)
        p = load_pathway_features(
            cohort,
            g.sample_ids,
            expected_feature_names=dev_path.feature_names,
        )
        ext_gene[cohort] = g
        ext_path[cohort] = p

    outer = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=OUTER_RANDOM_STATE,
    )

    rows = []
    paired = []
    selected = []
    counts = Counter()
    symbols = {}

    for fold, (tr, va) in enumerate(outer.split(dev.X, dev.y), start=1):
        print(f"\nOuter fold {fold}/{OUTER_SPLITS}")

        gs = make_search(gene=True)
        ps = make_search(gene=False)

        gs.fit(dev.X[tr], dev.y[tr])
        ps.fit(dev_path.X[tr], dev.y[tr])

        gm = gs.best_estimator_
        pm = ps.best_estimator_

        rows.append(metrics_row(
            fold, "gene", "GSE98793_outer_valid",
            len(tr), len(va), gs, dev.y[va],
            gm.predict_proba(dev.X[va])[:, 1]
        ))
        rows.append(metrics_row(
            fold, "pathway", "GSE98793_outer_valid",
            len(tr), len(va), ps, dev.y[va],
            pm.predict_proba(dev_path.X[va])[:, 1]
        ))

        selector = gm.named_steps["select"]
        if not isinstance(selector, SelectKBest):
            raise TypeError("Gene selector is not SelectKBest")
        idx = np.flatnonzero(selector.get_support())
        scores = np.asarray(selector.scores_, dtype=float)
        ranked = idx[np.argsort(scores[idx])[::-1]]

        for rank, j in enumerate(ranked, start=1):
            gid = dev.gene_ids[j]
            sym = dev.gene_symbols[j]
            counts[gid] += 1
            symbols[gid] = sym
            selected.append({
                "fold": fold,
                "rank": rank,
                "entrez_gene_id": gid,
                "gene_symbol": sym,
                "anova_f_score": float(scores[j]),
            })

        for cohort in EXTERNAL:
            g = ext_gene[cohort]
            p = ext_path[cohort]
            gr = metrics_row(
                fold, "gene", cohort, len(tr), len(g.y), gs, g.y,
                gm.predict_proba(g.X)[:, 1]
            )
            pr = metrics_row(
                fold, "pathway", cohort, len(tr), len(g.y), ps, g.y,
                pm.predict_proba(p.X)[:, 1]
            )
            rows += [gr, pr]
            paired.append({
                "fold": fold,
                "cohort": cohort,
                "gene_auroc": gr["auroc"],
                "pathway_auroc": pr["auroc"],
                "pathway_minus_gene": pr["auroc"] - gr["auroc"],
            })
            print(
                f"  {cohort}: gene={gr['auroc']:.3f}, "
                f"pathway={pr['auroc']:.3f}, "
                f"delta={pr['auroc'] - gr['auroc']:+.3f}"
            )

    write_csv(RESULTS_DIR / "fold_metrics.csv", rows)
    write_csv(RESULTS_DIR / "paired_external_auroc.csv", paired)
    write_csv(RESULTS_DIR / "selected_genes_by_fold.csv", selected)

    freq = [{
        "entrez_gene_id": gid,
        "gene_symbol": symbols[gid],
        "fold_count": n,
        "fold_fraction": n / OUTER_SPLITS,
    } for gid, n in counts.items()]
    freq.sort(key=lambda r: (-r["fold_count"], r["entrez_gene_id"]))
    write_csv(RESULTS_DIR / "selected_gene_frequency.csv", freq)

    summary = {
        "design": {
            "outer_splits": OUTER_SPLITS,
            "outer_random_state": OUTER_RANDOM_STATE,
            "inner_splits": INNER_SPLITS,
            "inner_random_state": INNER_RANDOM_STATE,
            "note": (
                "Each external estimate comes from a model trained on one "
                "original outer training fold (80% of GSE98793)."
            ),
        },
        "external_auroc": {},
        "paired_pathway_minus_gene": {},
        "gene_selection_stability": {
            "unique_genes_selected_across_folds": len(counts),
            "genes_selected_in_all_5_folds": sum(
                n == OUTER_SPLITS for n in counts.values()
            ),
        },
    }

    for cohort in EXTERNAL:
        summary["external_auroc"][cohort] = {}
        for model in ("gene", "pathway"):
            vals = [
                r["auroc"] for r in rows
                if r["evaluation"] == cohort and r["model"] == model
            ]
            summary["external_auroc"][cohort][model] = summarize(vals)

        vals = [
            r["pathway_minus_gene"] for r in paired
            if r["cohort"] == cohort
        ]
        summary["paired_pathway_minus_gene"][cohort] = summarize(vals)

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved reviewer training-variability results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
