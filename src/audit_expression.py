"""Reproducible QC summary of the three mapped GEO expression matrices.

Run from the project root after prepare_data.py and map_genes.py:
    python src/audit_expression.py

Reads the shared Entrez-gene matrices without changing or normalizing values.
Verifies gene order, sample IDs/labels, numeric values and duplicates.
Writes compact, Git-trackable diagnostic files to metadata/.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median

COHORTS = ("GSE98793", "GSE32280", "GSE39653")
PLATFORMS = {"GSE98793": "GPL570", "GSE32280": "GPL570", "GSE39653": "GPL10558"}


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear interpolation between nearest ranks (0 <= fraction <= 1)."""
    if not sorted_values:
        raise ValueError("Cannot compute percentile of an empty sample")
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (position - low) * (sorted_values[high] - sorted_values[low])


def selected_manifest(root: Path, cohort: str) -> list[dict[str, str]]:
    path = root / "metadata" / "cohort_manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = [r for r in csv.DictReader(stream)
                   if r["cohort"] == cohort and r["include"] == "yes"]
    if not records:
        raise ValueError(f"{cohort}: no selected samples in {path}")
    if any(r["label"] not in ("0", "1") for r in records):
        raise ValueError(f"{cohort}: selected samples have missing or invalid labels")
    return records


def inspect_cohort(root: Path, cohort: str, reference: list[str] | None):
    manifest = selected_manifest(root, cohort)
    expected_samples = [r["GSM"] for r in manifest]
    path = root / "data" / "processed" / f"{cohort}_shared_genes.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python src/map_genes.py first")

    sample_values: list[list[float]] = [[] for _ in expected_samples]
    gene_ids: list[str] = []
    seen_genes: set[str] = set()
    overall_min, overall_max = math.inf, -math.inf

    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader, None)
        if header is None or header[:2] != ["entrez_gene_id", "gene_symbol"]:
            raise ValueError(f"{cohort}: expected 'entrez_gene_id' and 'gene_symbol' columns")
        samples = header[2:]  # The first TWO columns are gene metadata, not samples.
        if samples != expected_samples:
            raise ValueError(f"{cohort}: sample IDs or order differ from the combined manifest")
        if len(samples) != len(set(samples)):
            raise ValueError(f"{cohort}: duplicate sample IDs")

        for line, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                raise ValueError(f"{cohort}: line {line} has {len(fields)} columns; expected {len(header)}")
            gene_id, symbol = fields[:2]
            if not gene_id.isdecimal() or not symbol:
                raise ValueError(f"{cohort}: invalid gene ID/symbol on line {line}")
            if gene_id in seen_genes:
                raise ValueError(f"{cohort}: duplicate Entrez Gene ID {gene_id}")
            seen_genes.add(gene_id)
            gene_ids.append(gene_id)
            for i, raw in enumerate(fields[2:]):
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"{cohort}: nonnumeric value at line {line}, sample {samples[i]}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{cohort}: nonfinite value at line {line}, sample {samples[i]}")
                sample_values[i].append(value)
                overall_min = min(overall_min, value)
                overall_max = max(overall_max, value)

    if not gene_ids:
        raise ValueError(f"{cohort}: matrix contains no genes")
    if reference is not None and gene_ids != reference:
        raise ValueError(f"{cohort}: shared Entrez Gene IDs or row order differ from other cohorts")

    sample_rows: list[dict] = []
    for record, values in zip(manifest, sample_values):
        values.sort()
        q25 = percentile(values, .25)
        q50 = percentile(values, .50)
        q75 = percentile(values, .75)
        sample_rows.append({
            "cohort": cohort,
            "GSM": record["GSM"],
            "label": record["label"],
            "diagnosis": record["diagnosis"],
            "min": round(values[0], 6),
            "q25": round(q25, 6),
            "median": round(q50, 6),
            "q75": round(q75, 6),
            "iqr": round(q75 - q25, 6),
            "max": round(values[-1], 6),
        })

    medians = [r["median"] for r in sample_rows]
    iqrs = [r["iqr"] for r in sample_rows]
    summary = {
        "cohort": cohort,
        "platform": PLATFORMS[cohort],
        "samples": len(samples),
        "MDD": sum(r["label"] == "1" for r in manifest),
        "controls": sum(r["label"] == "0" for r in manifest),
        "shared_gene_count": len(gene_ids),
        "overall_min": round(overall_min, 6),
        "overall_max": round(overall_max, 6),
        "median_of_sample_medians": round(median(medians), 6),
        "range_sample_medians": [round(min(medians), 6), round(max(medians), 6)],
        "median_within_sample_iqr": round(median(iqrs), 6),
        "range_within_sample_iqr": [round(min(iqrs), 6), round(max(iqrs), 6)],
    }
    return gene_ids, sample_rows, summary


def main():
    parser = argparse.ArgumentParser(description="Audit expression distributions for three shared-gene GEO matrices")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Project root (default: parent of src/)")
    args = parser.parse_args()
    root = args.root.resolve()
    reference: list[str] | None = None
    all_samples: list[dict] = []
    summaries: list[dict] = []

    for cohort in COHORTS:
        ids, sample_rows, summary = inspect_cohort(root, cohort, reference)
        if reference is None:
            reference = ids
        all_samples.extend(sample_rows)
        summaries.append(summary)
        lo, hi = summary["range_sample_medians"]
        print(f"{cohort} ({summary['platform']}): {summary['samples']} samples, {summary['shared_gene_count']:,} genes")
        print(f"  sample medians: {lo:.3f}–{hi:.3f}; median within-sample IQR: {summary['median_within_sample_iqr']:.3f}")
        print(f"  overall expression range: {summary['overall_min']:.3f}–{summary['overall_max']:.3f}")

    metadata = root / "metadata"
    metadata.mkdir(exist_ok=True)
    csv_path = metadata / "expression_qc_samples.csv"
    json_path = metadata / "expression_qc_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["cohort", "GSM", "label", "diagnosis", "min", "q25", "median", "q75", "iqr", "max"])
        writer.writeheader()
        writer.writerows(all_samples)
    report = {
        "method": "Unmodified expression values; per-sample quartiles interpolated by (n-1)*fraction; IQR=Q75-Q25",
        "note": "Descriptive diagnostic only. Does not establish normalization equivalence, fix platform shifts, or train a model.",
        "gene_ids_and_order_match": True,
        "cohorts": summaries,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Gene IDs and order match across all three cohorts.")
    print("Saved metadata/expression_qc_samples.csv and metadata/expression_qc_summary.json")
    print("Next: examine distributions and normalization provenance before model fitting.")


if __name__ == "__main__":
    main()
