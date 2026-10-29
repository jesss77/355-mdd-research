"""Load validated shared-gene matrices for modeling.

Default: development cohort only (GSE98793).
No normalization, feature selection, tuning, or model fitting happens here.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = "GSE98793"
EXTERNAL = ("GSE32280", "GSE39653")
ALL = (DEVELOPMENT, *EXTERNAL)


@dataclass(frozen=True)
class CohortData:
    cohort: str
    X: np.ndarray
    y: np.ndarray
    sample_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]
    gene_symbols: tuple[str, ...]


def _selected_manifest(root: Path, cohort: str) -> list[dict[str, str]]:
    path = root / "metadata" / "cohort_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run prepare_data.py first.")

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["cohort"] == cohort and r["include"] == "yes"
        ]

    if not rows:
        raise ValueError(f"{cohort}: no selected samples")
    if len({r["GSM"] for r in rows}) != len(rows):
        raise ValueError(f"{cohort}: duplicate GSM IDs")
    if any(r["label"] not in {"0", "1"} for r in rows):
        raise ValueError(f"{cohort}: invalid labels")

    return rows


def load_cohort(cohort: str = DEVELOPMENT, root: Path = ROOT) -> CohortData:
    if cohort not in ALL:
        raise ValueError(f"Unknown cohort {cohort!r}")

    manifest = _selected_manifest(root, cohort)
    sample_ids = tuple(r["GSM"] for r in manifest)
    y = np.asarray([int(r["label"]) for r in manifest], dtype=np.int8)

    path = root / "data" / "processed" / f"{cohort}_shared_genes.tsv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run map_genes.py first.")

    gene_ids, symbols, rows = [], [], []
    seen = set()

    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        if header is None or header[:2] != ["entrez_gene_id", "gene_symbol"]:
            raise ValueError(f"{cohort}: unexpected matrix header")
        if tuple(header[2:]) != sample_ids:
            raise ValueError(f"{cohort}: matrix sample IDs/order != manifest")

        for line_no, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                raise ValueError(f"{cohort}: width mismatch on line {line_no}")

            gid, symbol = fields[:2]
            if not gid.isdecimal() or not symbol:
                raise ValueError(f"{cohort}: invalid gene metadata on line {line_no}")
            if gid in seen:
                raise ValueError(f"{cohort}: duplicate gene {gid}")

            try:
                values = [float(v) for v in fields[2:]]
            except ValueError as exc:
                raise ValueError(f"{cohort}: nonnumeric value on line {line_no}") from exc

            if not all(math.isfinite(v) for v in values):
                raise ValueError(f"{cohort}: nonfinite value on line {line_no}")

            seen.add(gid)
            gene_ids.append(gid)
            symbols.append(symbol)
            rows.append(values)

    if not rows:
        raise ValueError(f"{cohort}: empty matrix")

    # Stored as genes x samples; modeling uses samples x genes.
    X = np.asarray(rows, dtype=np.float64).T

    if X.shape != (len(sample_ids), len(gene_ids)):
        raise AssertionError(f"{cohort}: unexpected X shape {X.shape}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"{cohort}: both classes are required")

    return CohortData(
        cohort=cohort,
        X=X,
        y=y,
        sample_ids=sample_ids,
        gene_ids=tuple(gene_ids),
        gene_symbols=tuple(symbols),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=ALL, default=DEVELOPMENT)
    args = parser.parse_args()

    data = load_cohort(args.cohort)
    print(f"{data.cohort}: {data.X.shape[0]} samples x {data.X.shape[1]:,} genes")
    print(f"labels: {(data.y == 1).sum()} MDD / {(data.y == 0).sum()} controls")
    print(f"finite values: {np.isfinite(data.X).all()}")
    print("transformations applied: NONE")

    if args.cohort in EXTERNAL:
        print("WARNING: external cohort loaded; do not use it for model selection.")


if __name__ == "__main__":
    main()
