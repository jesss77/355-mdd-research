"""Post-review random gene-set control."""

from __future__ import annotations

import argparse
import csv
import json
from typing import cast

import gseapy
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.model_data import DEVELOPMENT, EXTERNAL, ROOT, load_cohort
from src.modeling import classification_metrics
from src.nested_cv import (
    INNER_RANDOM_STATE,
    INNER_SPLITS,
    build_parameter_grid,
)
from src.pipeline_factory import build_model_pipeline

GMT_PATH = ROOT / "data" / "genesets" / "h.all.v2024.1.Hs.entrez.gmt"
RESULTS_DIR = ROOT / "results" / "reviewer_random_gene_sets"

RANDOM_SEED = 45
PFX = "ENTREZ_"
WEIGHT = 0.25
MIN_SIZE = 15
MAX_SIZE = 500

PATHWAY_SUMMARY = ROOT / "results" / "pathway" / "summary.json"
EXTERNAL_METRICS = ROOT / "results" / "external" / "metrics.csv"


def load_reference_results():
    pathway_summary = json.loads(
        PATHWAY_SUMMARY.read_text(encoding="utf-8")
    )
    hallmark_inner = float(
        pathway_summary["final_model"]["best_inner_auroc"]
    )

    hallmark_external = {}
    primary_gene_external = {}
    with EXTERNAL_METRICS.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        for row in csv.DictReader(f):
            if row["metric"] != "auroc":
                continue
            cohort = row["cohort"]
            model = row["model"]
            if cohort not in EXTERNAL:
                continue
            value = float(row["estimate"])
            if model == "pathway":
                hallmark_external[cohort] = value
            elif model == "gene":
                primary_gene_external[cohort] = value

    if set(hallmark_external) != set(EXTERNAL):
        raise ValueError("Missing committed pathway external AUROC")
    if set(primary_gene_external) != set(EXTERNAL):
        raise ValueError("Missing committed gene external AUROC")

    return hallmark_inner, hallmark_external, primary_gene_external


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-panels", type=int, default=100)
    p.add_argument("--batch-panels", type=int, default=20)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--jobs", type=int, default=-1)
    return p.parse_args()


def hallmark_sizes(shared_gene_ids):
    shared = set(shared_gene_ids)
    rows = []
    with GMT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            name = fields[0]
            size = sum(g in shared for g in fields[2:])
            rows.append((name, size))
    if len(rows) != 50:
        raise ValueError(f"Expected 50 Hallmark sets, found {len(rows)}")
    if any(not (MIN_SIZE <= s <= MAX_SIZE) for _, s in rows):
        raise ValueError("A Hallmark effective size is outside ssGSEA limits")
    return rows


def make_random_panels(shared_gene_ids, templates, n_panels):
    rng = np.random.default_rng(RANDOM_SEED)
    universe = np.asarray(shared_gene_ids, dtype=object)

    panel_names = {}
    gene_sets = {}
    mapping = []

    for panel in range(1, n_panels + 1):
        for i, (hallmark, size) in enumerate(templates, start=1):
            name = f"P{panel:03d}_S{i:02d}"
            genes = [
                str(g) for g in rng.choice(
                    universe, size=size, replace=False
                )
            ]
            panel_names.setdefault(panel, []).append(name)
            gene_sets[name] = [PFX + g for g in genes]
            mapping.append({
                "panel": panel,
                "set_index": i,
                "random_set": name,
                "matched_hallmark": hallmark,
                "effective_size": size,
                "entrez_gene_ids": ";".join(genes),
            })
    return panel_names, gene_sets, mapping


def expr_frame(data):
    return pd.DataFrame(
        data.X.T,
        index=[PFX + g for g in data.gene_ids],
        columns=list(data.sample_ids),
        dtype=float,
    )


def score(expr, panel_names, gene_sets, batch_panels, threads):
    panels = sorted(panel_names)
    pieces = []

    for start in range(0, len(panels), batch_panels):
        batch = panels[start:start + batch_panels]
        names = [
            n for panel in batch for n in panel_names[panel]
        ]
        gs = {n: gene_sets[n] for n in names}

        print(
            f"    ssGSEA panels {batch[0]}-{batch[-1]} "
            f"({len(names)} sets)"
        )
        r = gseapy.ssgsea(
            data=expr,
            gene_sets=gs,
            outdir=None,
            sample_norm_method="rank",
            correl_norm_type="rank",
            weight=WEIGHT,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            threads=threads,
            no_plot=True,
        )
        s = (
            r.res2d
            .pivot(index="Name", columns="Term", values="ES")
            .astype(float)
            .loc[list(expr.columns), names]
        )
        if s.isna().any().any():
            raise ValueError("NaN in random ssGSEA scores")
        pieces.append(s)

    return pd.concat(pieces, axis=1)


def make_search(jobs):
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=INNER_RANDOM_STATE,
    )
    return GridSearchCV(
        build_model_pipeline(selector="passthrough"),
        build_parameter_grid(),
        scoring="roc_auc",
        cv=cv,
        refit=True,
        n_jobs=jobs,
        return_train_score=False,
        error_score="raise",
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize(vals):
    a = np.asarray(vals, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def hallmark_position(random_vals, hallmark_val):
    a = np.asarray(random_vals, dtype=float)
    ge = int(np.sum(a >= hallmark_val))
    return {
        "hallmark_value": float(hallmark_val),
        "random_ge_hallmark_count": ge,
        "n_random": len(a),
        "hallmark_percentile_pct": float(
            100.0 * np.mean(a <= hallmark_val)
        ),
        "corrected_exceedance_fraction": float(
            (ge + 1) / (len(a) + 1)
        ),
    }


def main():
    args = parse_args()
    if args.n_panels <= 0 or args.batch_panels <= 0:
        raise ValueError("n-panels and batch-panels must be > 0")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    hallmark_inner, hallmark_external, primary_gene_external = (
        load_reference_results()
    )

    cohorts = {
        c: load_cohort(c)
        for c in (DEVELOPMENT, *EXTERNAL)
    }
    dev = cohorts[DEVELOPMENT]
    for c in EXTERNAL:
        if cohorts[c].gene_ids != dev.gene_ids:
            raise ValueError(f"{c}: shared genes/order differ")

    templates = hallmark_sizes(dev.gene_ids)
    print(
        "Hallmark effective sizes: "
        f"min={min(s for _, s in templates)}, "
        f"median={np.median([s for _, s in templates]):.1f}, "
        f"max={max(s for _, s in templates)}"
    )

    panel_names, gene_sets, mapping = make_random_panels(
        dev.gene_ids, templates, args.n_panels
    )
    write_csv(RESULTS_DIR / "random_gene_sets.csv", mapping)

    print(
        f"Generated {args.n_panels} panels / "
        f"{args.n_panels * 50} random matched sets "
        f"with seed {RANDOM_SEED}"
    )

    scores = {}
    for cohort, data in cohorts.items():
        print(f"\nScoring {cohort}...")
        scores[cohort] = score(
            expr_frame(data),
            panel_names,
            gene_sets,
            args.batch_panels,
            args.threads,
        )

    results = []
    print("\nTraining/evaluating random panels...")
    for panel in range(1, args.n_panels + 1):
        names = panel_names[panel]
        search = make_search(args.jobs)
        search.fit(
            scores[DEVELOPMENT][names].to_numpy(float),
            dev.y,
        )
        model = cast(Pipeline, search.best_estimator_)

        row = {
            "panel": panel,
            "best_inner_auroc": float(search.best_score_),
            "C": float(search.best_params_["model__C"]),
            "l1_ratio": float(
                search.best_params_["model__l1_ratio"]
            ),
        }

        for cohort in EXTERNAL:
            data = cohorts[cohort]
            prob = model.predict_proba(
                scores[cohort][names].to_numpy(float)
            )[:, 1]
            m = classification_metrics(data.y, prob)
            row[f"{cohort}_auroc"] = m["auroc"]
            row[f"{cohort}_f1"] = m["f1"]
            row[f"{cohort}_sensitivity"] = m["sensitivity"]
            row[f"{cohort}_specificity"] = m["specificity"]
            row[f"{cohort}_minus_primary_gene_auroc"] = (
                m["auroc"] - primary_gene_external[cohort]
            )

        results.append(row)

        if panel == 1 or panel % 5 == 0 or panel == args.n_panels:
            print(
                f"  panel {panel:03d}/{args.n_panels}: "
                f"inner={row['best_inner_auroc']:.3f}, "
                f"GSE32280={row['GSE32280_auroc']:.3f}, "
                f"GSE39653={row['GSE39653_auroc']:.3f}"
            )

    write_csv(RESULTS_DIR / "panel_results.csv", results)

    inner = [r["best_inner_auroc"] for r in results]
    summary = {
        "settings": {
            "n_panels": args.n_panels,
            "sets_per_panel": 50,
            "random_seed": RANDOM_SEED,
            "matching": (
                "Each random set matches the number of genes from its "
                "Hallmark template present in the 16,467-gene shared universe."
            ),
            "sampling": (
                "Without replacement within a set; "
                "overlap across sets is allowed."
            ),
            "ssgsea": {
                "tool": f"gseapy {gseapy.__version__}",
                "score": "raw ES",
                "sample_norm_method": "rank",
                "correl_norm_type": "rank",
                "weight": WEIGHT,
                "min_size": MIN_SIZE,
                "max_size": MAX_SIZE,
            },
            "model_selection": (
                "Same 5-fold development-only inner GridSearchCV and "
                "elastic-net grid used for the final Hallmark model."
            ),
        },
        "development_best_inner_auroc": {
            "random_distribution": summarize(inner),
            "hallmark_position": hallmark_position(
                inner, hallmark_inner
            ),
        },
        "external_auroc": {},
    }

    for cohort in EXTERNAL:
        vals = [r[f"{cohort}_auroc"] for r in results]
        summary["external_auroc"][cohort] = {
            "random_distribution": summarize(vals),
            "hallmark_position": hallmark_position(
                vals, hallmark_external[cohort]
            ),
            "primary_gene_reference": (
                primary_gene_external[cohort]
            ),
        }

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nHallmark positions:")
    print(
        "  development inner score:",
        summary["development_best_inner_auroc"]["hallmark_position"],
    )
    for cohort in EXTERNAL:
        print(
            f"  {cohort}:",
            summary["external_auroc"][cohort]["hallmark_position"],
        )
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
