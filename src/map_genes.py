"""Map selected GEO probe matrices to shared Entrez Gene IDs.

Inputs:
    data/annotation/GPL570.annot.gz
    data/annotation/GPL10558.annot.gz
    data/interim/GSE*_selected_probes.tsv.gz
    metadata/cohort_manifest.csv

Outputs (data/processed/ is intentionally Git-ignored):
    data/processed/GSE*_shared_genes.tsv.gz (genes in rows, samples in columns)
    metadata/gene_mapping_summary.json
    metadata/shared_genes.csv

Rules: keep only probes with a single numeric Entrez Gene ID and a single
nonblank gene symbol; collapse multiple probes for the same gene with the median
within each sample; retain genes measured in all three cohorts. No sample-wise
normalization or model fitting is performed here.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {"GSE98793": "GPL570", "GSE32280": "GPL570", "GSE39653": "GPL10558"}
IGNORE_SYMBOLS = {"", "---", "NA", "N/A", "NULL", "?"}


def load_annotation(platform: str) -> tuple[dict[str, tuple[str, str]], dict[str, int]]:
    path = ROOT / "data" / "annotation" / f"{platform}.annot.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Download the GEO platform annotation first.")
    annotations: dict[str, tuple[str, str]] = {}
    conflicts: set[str] = set()
    stats: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = (line for line in stream if line.strip() and not line.startswith(("#", "!", "^")))
        reader = csv.DictReader(rows, delimiter="\t")
        required = {"ID", "Gene ID", "Gene symbol"}
        if reader.fieldnames is None or not required.issubset({x.strip() for x in reader.fieldnames}):
            raise ValueError(f"{platform}: expected annotation columns {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            stats["annotation_rows"] += 1
            probe = (row.get("ID") or "").strip()
            gene_id = (row.get("Gene ID") or "").strip()
            symbol = (row.get("Gene symbol") or "").strip()
            if not probe:
                stats["missing_probe_id"] += 1
                continue
            if not re.fullmatch(r"[0-9]+", gene_id) or "///" in symbol or symbol.upper() in IGNORE_SYMBOLS:
                stats["unmapped_or_ambiguous"] += 1
                continue
            pair = (gene_id, symbol)
            if probe in conflicts:
                continue
            if probe in annotations and annotations[probe] != pair:
                annotations.pop(probe)
                conflicts.add(probe)
                stats["conflicting_duplicate_probe"] += 1
            else:
                annotations[probe] = pair
    stats["mapped_unambiguous_probes"] = len(annotations)
    stats["unique_entrez_genes_in_annotation"] = len({g for g, _ in annotations.values()})
    return annotations, dict(stats)


def selected_sample_ids(cohort: str) -> list[str]:
    path = ROOT / "metadata" / "cohort_manifest.csv"
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return [row["GSM"] for row in csv.DictReader(file)
                if row["cohort"] == cohort and row["include"] == "yes"]


def map_cohort(cohort: str, annotations: dict[str, tuple[str, str]]) -> tuple[list[str], dict[str, list[list[float]]], dict[str, str], dict[str, int]]:
    path = ROOT / "data" / "interim" / f"{cohort}_selected_probes.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run python src/prepare_data.py first.")
    gene_probes: dict[str, list[list[float]]] = defaultdict(list)
    gene_symbols: dict[str, str] = {}
    stats: Counter[str] = Counter()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        header = next(reader)
        if not header or header[0] != "ID_REF" or header[1:] != selected_sample_ids(cohort):
            raise ValueError(f"{cohort}: sample columns do not match the selected manifest in order")
        sample_ids = header[1:]
        if not sample_ids:
            raise ValueError(f"{cohort}: no sample columns")
        for fields in reader:
            if len(fields) != len(header):
                raise ValueError(f"{cohort}: malformed expression row {stats['probe_rows'] + 1}")
            stats["probe_rows"] += 1
            mapped = annotations.get(fields[0])
            if mapped is None:
                stats["excluded_unmapped_probes"] += 1
                continue
            gid, symbol = mapped
            try:
                vals = [float(s) for s in fields[1:]]
            except ValueError as exc:
                raise ValueError(f"{cohort}: non-numeric values for {fields[0]}") from exc
            gene_probes[gid].append(vals)
            gene_symbols.setdefault(gid, symbol)
            stats["mapped_probe_rows"] += 1
    stats["unique_entrez_genes"] = len(gene_probes)
    stats["genes_with_multiple_probes"] = sum(len(p) > 1 for p in gene_probes.values())
    return sample_ids, gene_probes, gene_symbols, dict(stats)


def write_shared_cohort(cohort: str, sample_ids: list[str], gene_probes: dict[str, list[list[float]]], symbols: dict[str, str], shared: list[str]) -> None:
    path = ROOT / "data" / "processed" / f"{cohort}_shared_genes.tsv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["entrez_gene_id", "gene_symbol", *sample_ids])
        for gid in shared:
            probes = gene_probes[gid]
            values = [median(probe[i] for probe in probes) for i in range(len(sample_ids))]
            writer.writerow([gid, symbols[gid], *[f"{v:.8g}" for v in values]])


def main() -> None:
    annotations = {}
    annotation_reports = {}
    for platform in sorted(set(COHORTS.values())):
        annotations[platform], annotation_reports[platform] = load_annotation(platform)
        print(f"{platform}: {annotation_reports[platform]['mapped_unambiguous_probes']:,} unambiguous probe mappings, "
              f"{annotation_reports[platform]['unique_entrez_genes_in_annotation']:,} genes")
    cohorts = {}
    for cohort, platform in COHORTS.items():
        samples, genes, symbols, stats = map_cohort(cohort, annotations[platform])
        cohorts[cohort] = (samples, genes, symbols, stats)
        print(f"{cohort}: {len(samples)} samples, {stats['mapped_probe_rows']:,}/{stats['probe_rows']:,} "
              f"probes mapped, {stats['unique_entrez_genes']:,} genes")
    shared = sorted(set.intersection(*(set(cohorts[c][1]) for c in COHORTS)), key=int)
    if not shared:
        raise ValueError("No shared Entrez Gene IDs among the three cohorts")
    print(f"Shared genes across all three cohorts: {len(shared):,}")
    symbol_disagreements = 0
    shared_path = ROOT / "metadata" / "shared_genes.csv"
    with shared_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entrez_gene_id", "gene_symbol_GPL570", "gene_symbol_GPL10558"])
        for gid in shared:
            affy_symbol = cohorts["GSE98793"][2][gid]
            illumina_symbol = cohorts["GSE39653"][2][gid]
            symbol_disagreements += affy_symbol != illumina_symbol
            writer.writerow([gid, affy_symbol, illumina_symbol])
    for cohort, (samples, genes, symbols, _) in cohorts.items():
        write_shared_cohort(cohort, samples, genes, symbols, shared)
        print(f"  wrote data/processed/{cohort}_shared_genes.tsv.gz")
    report = {
        "mapping_rule": "Single Entrez Gene ID and single symbol per probe; exclude ambiguous/unmapped; median of probes per gene per sample",
        "note": "Gene IDs matched across platforms; expression values are NOT yet scale-harmonized or safe for cross-platform training.",
        "annotations": annotation_reports,
        "cohorts": {c: {"platform": COHORTS[c], "samples": len(v[0]), **v[3]} for c, v in cohorts.items()},
        "shared_genes": len(shared),
        "shared_gene_symbol_disagreements": symbol_disagreements,
    }
    out = ROOT / "metadata" / "gene_mapping_summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {shared_path.relative_to(ROOT)} and {out.relative_to(ROOT)}")
    print("Next: evaluate expression distributions and preprocessing; do NOT train until checked.")


if __name__ == "__main__":
    main()
