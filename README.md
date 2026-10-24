# CMPS 355 — MDD Transcriptomics Research

Research project for CMPS 355 (Bioinformatics Algorithms) at AUB.

## Overview
We're investigating whether pathway-level representations improve cross-cohort generalization in Major Depressive Disorder (MDD) classification compared to individual gene-expression features.

The experiment compares both representations using the same classifier and evaluates them on independent cohorts.

## Research question

Do pathway-level transcriptomic representations improve cross-cohort generalization in MDD classification compared with gene-level representations?

> We will compare gene-level features against Hallmark pathway scores, using the same elastic-net logistic regression approach and testing on independent cohorts.

## Datasets

| Dataset | Purpose | Selected samples |
| --- | --- | --- |
| GSE98793 | Model development | 128 MDD, 64 controls |
| GSE32280 | External validation | 8 pretreatment MDD, 8 controls |
| GSE39653 | External validation | 21 MDD, 24 controls |

Data are from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/).

## Reproducing data preparation

From the project root, using Python 3.14:

```bash
python src/prepare_data.py
```

Download the two GEO platform annotations (PowerShell):

```powershell
New-Item -ItemType Directory -Force data\annotation
Invoke-WebRequest -Uri "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz" -OutFile "data\annotation\GPL570.annot.gz"
Invoke-WebRequest -Uri "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL10nnn/GPL10558/annot/GPL10558.annot.gz" -OutFile "data\annotation\GPL10558.annot.gz"
```

Then run:

```bash
python src/map_genes.py
python src/audit_expression.py
```

The pipeline produces:
- `metadata/cohort_manifest.csv`: sample selection across all three studies.
- `metadata/dataset_summary.json`: initial dataset checks.
- `metadata/shared_genes.csv` and `metadata/gene_mapping_summary.json`: gene mapping and overlap.
- `metadata/expression_qc_samples.csv` and `metadata/expression_qc_summary.json`: per-sample expression distributions.
- `data/processed/`: selected, shared-gene expression matrices (generated locally; not committed).

## Progress

Completed sample selection and mapped platform-specific probes to **16,467 shared Entrez genes**, aggregating multiple probes per gene by their median expression. The expression-audit script checks sample distributions and matching gene order; we still need to interpret its results and verify cross-platform preprocessing.

Next: validate preprocessing compatibility, construct the 50-gene and 50-pathway representations, train using GSE98793 only, and evaluate the frozen models on both external datasets.

**Note:** The current matrices have not been harmonized for model training yet.
