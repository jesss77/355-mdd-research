"""CMPS 355 Phase 1: acquire, verify and select all three GEO cohorts.

Place this file in the EXISTING project's src/ folder and run:
    python src/prepare_data.py

Python standard library only; tested with Python 3.14. Produces ONE combined
metadata/cohort_manifest.csv, one dataset_summary.json, and separate selected
probe-level matrices. These matrices are NOT gene-mapped or harmonized and are
NOT yet ready for cross-platform machine learning.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("GSE98793", "GSE32280", "GSE39653")
PLATFORMS = {"GSE98793": "GPL570", "GSE32280": "GPL570", "GSE39653": "GPL10558"}
EXPECTED = {
    "GSE98793": (192, 128, 64, 0),
    "GSE32280": (37, 8, 8, 21),
    "GSE39653": (53, 21, 24, 8),
}
FIELDS = ["cohort", "GSM", "role", "platform", "title", "diagnosis", "label",
          "include", "reason", "sample_type", "treatment", "anxiety", "batch", "sex", "age", "warning"]


def geo_url(cohort: str) -> str:
    return ("https://ftp.ncbi.nlm.nih.gov/geo/series/"
            f"{cohort[:-3]}nnn/{cohort}/matrix/{cohort}_series_matrix.txt.gz")


def fetch(cohort: str) -> Path:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{cohort}_series_matrix.txt.gz"
    if path.exists() and path.stat().st_size:
        print(f"{cohort}: existing download, no repeat")
        return path
    tmp = path.with_name(path.name + ".partial")
    print(f"{cohort}: downloading GEO series matrix...")
    try:
        with urlopen(geo_url(cohort), timeout=90) as response, tmp.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for block in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tabbed(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))


def field_map(s: str) -> dict[str, str]:
    result = {}
    for item in s.split(" | "):
        if not item:
            continue
        if ": " not in item:
            raise ValueError(f"Unexpected characteristics field {item!r}")
        key, value = item.split(": ", 1)
        key = key.lower().strip()
        if key in result:
            raise ValueError(f"Duplicate characteristic {key!r}")
        result[key] = value.strip()
    return result


def choose(cohort: str, original: dict[str, str]) -> dict[str, str]:
    title = original["title"].strip()
    attrs = field_map(original["characteristics"])
    sample_type, timepoint, reason, warning = "", "", "", ""
    include, diagnosis, anxiety = True, "", attrs.get("anxiety", "")

    if cohort == "GSE98793":
        source = original["source_name"].lower()
        group = attrs.get("subject group", "")
        if group == "CASE; major depressive disorder (MDD) patient" and title.startswith("whole_blood_case_") and source == "whole_blood_case":
            diagnosis = "MDD"
        elif group == "CNTL; healthy control" and title.startswith("whole_blood_control_") and source == "whole_blood_control":
            diagnosis = "control"
        else:
            raise ValueError(f"Conflicting or unknown label: {title!r}; {group!r}")
        if attrs.get("tissue", "").lower() != "whole blood" or anxiety not in {"yes", "no"}:
            raise ValueError("Unexpected tissue/anxiety metadata")
        sample_type = "whole blood"

    elif cohort == "GSE32280":
        disease = attrs.get("disease status", "")
        sample_type = attrs.get("cell type", "").lower()
        if sample_type != "lymphocytes":
            raise ValueError(f"Expected lymphocytes, got {sample_type!r}")
        if disease == "healthy control":
            diagnosis = "control"
            if not re.fullmatch(r"(?:health|healthy) control \d+", title, re.I):
                raise ValueError("Healthy-control title does not match disease status")
        elif disease in ("MDD patient", "SSD patient"):
            diagnosis = "MDD" if disease == "MDD patient" else "SSD"
            if not re.fullmatch(rf"{diagnosis} patient \d+ (before treatment|after treatment with Venlafaxine)", title, re.I):
                raise ValueError("Case title does not match disease status/treatment")
        else:
            raise ValueError(f"Unknown disease status {disease!r}")
        timepoint = attrs.get("treatment", "")
        if diagnosis != "control":
            if timepoint not in {"before treatment", "after treatment with Venlafaxine"}:
                raise ValueError(f"Unknown timepoint {timepoint!r}")
            if ("before treatment" in title) != (timepoint == "before treatment"):
                raise ValueError("Title and treatment timepoint disagree")
        if diagnosis == "SSD":
            include, reason = False, "SSD, not MDD"
            if "MDD patient" in original.get("description", ""):
                warning = "SSD free-text description says MDD; disease status/title used"
        elif diagnosis == "MDD" and timepoint != "before treatment":
            include, reason = False, "post-treatment repeated measurement"
    else:  # GSE39653, clinical labels named in GEO sample titles.
        m = re.fullmatch(r"(MDD|HC|BD1|BD2)-(\d+)", title, re.I)
        if not m:
            raise ValueError(f"Unexpected GSE39653 sample title {title!r}")
        prefix = m.group(1).upper()
        diagnosis = {"MDD": "MDD", "HC": "control", "BD1": "bipolar", "BD2": "bipolar"}[prefix]
        sample_type = attrs.get("tissue", original["source_name"]).strip()
        if "pbmc" not in sample_type.lower() and "pbmc" not in original["source_name"].lower():
            raise ValueError(f"Expected PBMC, got {sample_type!r}")
        observed = attrs.get("disease", attrs.get("disease status", "")).lower()
        if observed:
            matches = {
                "MDD": ("major depress" in observed or "mdd" in observed),
                "control": ("healthy" in observed or "control" in observed),
                "bipolar": ("bipolar" in observed or observed.startswith("bd")),
            }
            if not matches[diagnosis]:
                raise ValueError(f"GSE39653 title/disease contradiction {title!r} vs {observed!r}")
        if diagnosis == "bipolar":
            include, reason = False, "bipolar disorder, not MDD"

    return {
        "cohort": cohort, "GSM": original["GSM"],
        "role": "development" if cohort == "GSE98793" else "external_test",
        "platform": original["platform_id"], "title": title, "diagnosis": diagnosis,
        "label": "1" if include and diagnosis == "MDD" else "0" if include and diagnosis == "control" else "",
        "include": "yes" if include else "no", "reason": reason,
        "sample_type": sample_type, "treatment": timepoint, "anxiety": anxiety,
        "batch": attrs.get("batch", ""), "sex": attrs.get("gender", attrs.get("sex", "")),
        "age": attrs.get("age", ""), "warning": warning,
    }


def parse_metadata_from_matrix(cohort: str, raw: Path) -> tuple[list[dict[str, str]], list[str]]:
    per_key = defaultdict(list)
    with gzip.open(raw, "rt", encoding="utf-8", errors="replace") as inp:
        for line in inp:
            line = line.rstrip("\r\n")
            if line == "!series_matrix_table_begin":
                break
            if line.startswith("!Sample_"):
                key, sep, remainder = line.partition("\t")
                if sep:
                    per_key[key[len("!Sample_"):]].append(tabbed(remainder))
    ids_rows = per_key.get("geo_accession", [])
    if len(ids_rows) != 1:
        raise ValueError(f"{cohort}: missing/duplicated GEO sample-ID row")
    ids = ids_rows[0]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{cohort}: duplicate GSM identifiers")
    for key, blocks in per_key.items():
        if any(len(block) != len(ids) for block in blocks):
            raise ValueError(f"{cohort}: {key} metadata width mismatch")
    records = []
    for i, gsm in enumerate(ids):
        def val(key: str) -> str:
            return " | ".join(row[i] for row in per_key.get(key, []))
        records.append({"GSM": gsm, "title": val("title"), "source_name": val("source_name_ch1"),
                        "characteristics": val("characteristics_ch1"), "description": val("description"),
                        "platform_id": val("platform_id")})
    return records, ids


def read_existing_metadata(cohort: str) -> list[dict[str, str]]:
    # Only for local tests or for investigating labels before the third download.
    file = ROOT / "metadata" / f"{cohort}_sample_metadata.csv"
    with file.open(newline="", encoding="utf-8-sig") as fh:
        records = list(csv.DictReader(fh))
    return records


def validate_labels(cohort: str, originals: list[dict[str, str]]) -> list[dict[str, str]]:
    count, cases, controls, excluded = EXPECTED[cohort]
    if len(originals) != count:
        raise ValueError(f"{cohort}: {len(originals)} records; expected {count}")
    if len({r['GSM'] for r in originals}) != len(originals):
        raise ValueError(f"{cohort}: repeated GSM identifier")
    results = []
    for rec in originals:
        if rec.get("platform_id") != PLATFORMS[cohort]:
            raise ValueError(f"{cohort}/{rec['GSM']}: unexpected platform {rec.get('platform_id')!r}")
        try:
            results.append(choose(cohort, rec))
        except ValueError as exc:
            raise ValueError(f"{cohort}/{rec['GSM']}: {exc}") from exc
    selected = Counter(r["diagnosis"] for r in results if r["include"] == "yes")
    excluded_actual = sum(r["include"] == "no" for r in results)
    if (selected["MDD"], selected["control"], excluded_actual) != (cases, controls, excluded):
        raise ValueError(f"{cohort}: selection count mismatch: {dict(selected)}, excluded={excluded_actual}")
    if cohort == "GSE32280":
        titles = [r["title"] for r in originals]
        pairs = []
        for row, title in zip(results, titles):
            if row["include"] != "yes":
                continue
            participant = re.search(r"\d+", title)
            if participant is None:
                raise ValueError(f"GSE32280: no participant number in title {title!r}")
            pairs.append((row["diagnosis"], participant.group()))
        if len(pairs) != len(set(pairs)):
            raise ValueError("GSE32280: repeated participant in selected cohort")
    return results


def export_selected_and_audit(cohort: str, raw: Path, ids: list[str], selection: list[dict[str, str]]) -> dict:
    output_dir = ROOT / "data" / "interim"
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{cohort}_selected_probes.tsv.gz"
    indices = [i for i, r in enumerate(selection) if r["include"] == "yes"]
    wanted = [ids[i] for i in indices]
    rows = 0
    invalid = 0
    minimum, maximum = math.inf, -math.inf
    seen_probes = set()
    duplicates = 0
    in_matrix, header_found = False, False
    tmp = dest.with_name(dest.name + ".partial")
    try:
        with gzip.open(raw, "rt", encoding="utf-8", errors="replace") as source, gzip.open(tmp, "wt", newline="", encoding="utf-8", compresslevel=5) as output:
            writer = csv.writer(output, delimiter="\t")
            for line in source:
                line = line.rstrip("\r\n")
                if line == "!series_matrix_table_begin":
                    in_matrix = True
                    continue
                if line == "!series_matrix_table_end":
                    in_matrix = False
                    break
                if not in_matrix:
                    continue
                fields = tabbed(line)
                if not header_found:
                    if fields[0] != "ID_REF" or fields[1:] != ids:
                        raise ValueError(f"{cohort}: expression columns do not match metadata IDs/order")
                    writer.writerow(["ID_REF"] + wanted)
                    header_found = True
                    continue
                if len(fields) != len(ids) + 1:
                    raise ValueError(f"{cohort}: row {rows + 1} width mismatch")
                if fields[0] in seen_probes:
                    duplicates += 1
                seen_probes.add(fields[0])
                rows += 1
                selected_values = [fields[i + 1] for i in indices]
                for item in selected_values:
                    try:
                        val = float(item)
                    except (ValueError, TypeError):
                        invalid += 1
                        continue
                    if not math.isfinite(val):
                        invalid += 1
                    else:
                        minimum = min(minimum, val)
                        maximum = max(maximum, val)
                writer.writerow([fields[0]] + selected_values)
        if not header_found or rows == 0:
            raise ValueError(f"{cohort}: expression table not found or empty")
        if invalid:
            raise ValueError(f"{cohort}: {invalid} invalid/non-finite values in selected subset")
        if duplicates:
            raise ValueError(f"{cohort}: {duplicates} repeated probe IDs")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"total_expression_probe_rows": rows, "selected_samples": len(indices),
            "selected_value_min": minimum, "selected_value_max": maximum,
            "numeric_values_checked": rows * len(indices), "invalid_values": invalid,
            "probe_file": str(dest.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all three CMPS355 GEO datasets in one command")
    parser.add_argument("--metadata-only", action="store_true", help="Validate existing per-cohort metadata CSVs without raw GEO files (for debugging)")
    args = parser.parse_args()
    all_rows = []
    summaries = []
    for cohort in COHORTS:
        if args.metadata_only and not (ROOT / "metadata" / f"{cohort}_sample_metadata.csv").exists():
            print(f"{cohort}: metadata CSV unavailable; skipped (--metadata-only)")
            continue
        if args.metadata_only:
            original = read_existing_metadata(cohort)
            ids = [r["GSM"] for r in original]
            raw = None
        else:
            raw = fetch(cohort)
            original, ids = parse_metadata_from_matrix(cohort, raw)
        selected = validate_labels(cohort, original)
        summary = {"cohort": cohort, "platform": PLATFORMS[cohort],
                   "total": len(selected), "included": sum(r["include"] == "yes" for r in selected),
                   "MDD": sum(r["label"] == "1" for r in selected),
                   "control": sum(r["label"] == "0" for r in selected),
                   "excluded": sum(r["include"] == "no" for r in selected),
                   "metadata_warnings": sum(bool(r["warning"]) for r in selected),
                   "source": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={cohort}"}
        if raw is not None:
            summary["download_sha256"] = checksum(raw)
            summary.update(export_selected_and_audit(cohort, raw, ids, selected))
        all_rows.extend(selected)
        summaries.append(summary)
        print(f"{cohort}: {summary['included']} selected: {summary['MDD']} MDD / {summary['control']} controls; excluded {summary['excluded']}")
        if raw is not None:
            print(f"  {summary['total_expression_probe_rows']} probes; selected values [{summary['selected_value_min']:.3f}, {summary['selected_value_max']:.3f}]")
        if summary["metadata_warnings"]:
            print(f"  {summary['metadata_warnings']} metadata warning(s), see combined manifest")
    if not all_rows:
        raise RuntimeError("Nothing to process")
    metadata = ROOT / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    with (metadata / "cohort_manifest.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_audits": summaries,
              "WARNING": "Selected matrices still contain PLATFORM-SPECIFIC PROBES; DO NOT train a cross-platform classifier until identifiers and preprocessing are validated."}
    (metadata / "dataset_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nONE manifest: metadata/cohort_manifest.csv")
    print("Audit summary: metadata/dataset_summary.json")

if __name__ == "__main__":
    main()
