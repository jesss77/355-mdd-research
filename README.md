# CMPS 355 — MDD Transcriptomics Research

Research project for CMPS 355 - Bioinformatics Algorithms.

## Overview

We're investigating whether pathway-level representations improve cross-cohort generalization in Major Depressive Disorder (MDD) classification compared to individual gene-expression features.

The experiment compares both representations using the same classifier and evaluates them on independent cohorts.

## Datasets

|  Dataset  |      Purpose         |       Samples        |
|-----------|----------------------|----------------------|
|  GSE98793 | Training/development | 128 MDD, 64 controls |
|  GSE32280 | External validation  | 8 MDD, 8 controls    |
|  GSE39653 | External validation  | 21 MDD, 24 controls  |

All datasets are publicly available through NCBI GEO.

## Current Progress

**Phase 1 — Dataset Preparation**

Completed dataset acquisition, sample selection, and initial validation. A single preprocessing script handles all three datasets, excludes irrelevant samples, and exports the selected expression matrices.

To reproduce this step:

```bash
python src/prepare_data.py
```

The script generates:
- `metadata/cohort_manifest.csv`
- `metadata/dataset_summary.json`
- Selected probe-level matrices in `data/interim/`

## Next Steps

Map probes to shared genes, verify preprocessing compatibility, construct gene-level and pathway-level features, and train and evaluate both models.

The current matrices have not been harmonized for model training yet.