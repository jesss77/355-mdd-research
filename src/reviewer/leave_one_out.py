from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "reviewer_leave_one_out"

pred = pd.read_csv(ROOT / "results/external/predictions.csv")
pred = pred[pred["cohort"] == "GSE32280"].reset_index(drop=True)

rows = []

for i in range(len(pred)):
    remaining = pred.drop(index=i)

    gene_auc = roc_auc_score(
        remaining["label"], remaining["gene_probability"]
    )
    pathway_auc = roc_auc_score(
        remaining["label"], remaining["pathway_probability"]
    )

    rows.append({
        "excluded_sample": pred.loc[i, "GSM"],
        "excluded_label": pred.loc[i, "label"],
        "gene_auroc": gene_auc,
        "pathway_auroc": pathway_auc,
        "paired_difference": pathway_auc - gene_auc,
    })

results = pd.DataFrame(rows)

OUT.mkdir(parents=True, exist_ok=True)
results.to_csv(OUT / "results.csv", index=False)

for col in ["gene_auroc", "pathway_auroc", "paired_difference"]:
    print(f"{col}: {results[col].min():.3f} to {results[col].max():.3f}")

print("Positive differences:",
      (results["paired_difference"] > 0).sum(),
      "/", len(results))