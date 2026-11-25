# Peer-review robustness analyses

These analyses were added after peer review and were not part of the original pre-specified experiment.

## Environment

Install the exact pinned environment:

```bash
python -m pip install -r requirements.txt
```

Run all commands from the repository root with the virtual environment active.

## Required processed matrices

The following Git-ignored files must exist for the training-variability and random gene-set analyses:

- data/processed/GSE98793_shared_genes.tsv.gz
- data/processed/GSE32280_shared_genes.tsv.gz
- data/processed/GSE39653_shared_genes.tsv.gz

They can be regenerated using the original data preparation and mapping pipeline.

## Training-variability analysis

Run:

```bash
python -m src.reviewer.training_variability
```

Outputs:

- results/reviewer_training_variability/fold_metrics.csv
- results/reviewer_training_variability/paired_external_auroc.csv
- results/reviewer_training_variability/selected_genes_by_fold.csv
- results/reviewer_training_variability/selected_gene_frequency.csv
- results/reviewer_training_variability/summary.json

This analysis refits the original gene and pathway pipelines in each of the five original outer training folds and applies each fitted model to both external cohorts.

## Random gene-set control

Smoke test:

```bash
python -m src.reviewer.random_gene_sets --n-panels 3 --batch-panels 3 --threads 4 --jobs -1
```

Full analysis:

```bash
python -m src.reviewer.random_gene_sets --n-panels 100 --batch-panels 20 --threads 4 --jobs -1
```

Random seed: 45.

Each random panel contains 50 random gene sets. Each set is matched to the effective size of one Hallmark set after restriction to the common 16,467-gene universe.

Sampling is without replacement within each random set. Overlap across random sets is allowed.

The same raw ssGSEA settings and elastic-net development-only model-selection procedure used for the Hallmark representation are applied to each random panel.

Outputs:

- results/reviewer_random_gene_sets/random_gene_sets.csv
- results/reviewer_random_gene_sets/panel_results.csv
- results/reviewer_random_gene_sets/summary.json

## Leave-one-out sensitivity analysis

Each of the 16 GSE32280 samples is excluded individually, and gene and pathway AUROCs are recalculated using the original frozen predictions.

No model retraining or hyperparameter selection is performed. This analysis does not require the processed expression matrices.

Input:

- results/external/predictions.csv

Run:

```bash
python -m src.reviewer.leave_one_out
```

Output:

- results/reviewer_leave_one_out/results.csv

Results:

- Gene AUROC range: 0.393–0.518
- Pathway AUROC range: 0.714–0.821
- Paired pathway-minus-gene AUROC difference: +0.196 to +0.357
- Positive paired differences: 16/16

The model ordering remains unchanged across all 16 single-sample omissions, although uncertainty due to the small external cohort remains.