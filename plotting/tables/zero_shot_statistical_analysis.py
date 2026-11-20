#!/usr/bin/env python3
"""
Statistical analysis for zero-shot cell type annotation results.
Computes all numbers needed for the results section and supplementary tables.

Outputs (in outputs/tables/):
  - zeroshot_per_dataset_metrics.csv       (9 datasets with metadata)
  - zeroshot_misclassification_patterns.csv (top confusion pairs per dataset)
  - zeroshot_per_celltype_metrics.csv       (per cell-type one-vs-rest metrics)
  - zeroshot_lineage_metrics.csv            (one-vs-rest broad lineage F1/Prec/Rec)
  - zeroshot_size_bin_metrics.csv           (one-vs-rest per fine type, averaged by bin)
"""

import os
import pathlib
import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(".")
DATA_ROOT = PROJECT_ROOT / "resources" / "scLEAP" / "results" / "tables"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TISSUE_MAP = {
    "E-MTAB-9221.project": "Blood",
    "GSE185948": "Kidney",
    "E-HCAD-15.project": "Lung",
    "E-MTAB-5061": "Pancreas",
    "GSE86473": "Pancreas",
    "E-HCAD-9.project": "Liver",
    "GSE84133": "Pancreas",
    "E-CURD-114.project": "Lung",
    "GSE81608": "Pancreas",
    "GSE111976": "Endometrium",
}

# Full broad_map from Dung's Figure2_plotting.ipynb Cell 11
BROAD_MAP = {
    # Immune cell
    "b cell": "Immune cell", "t cell": "Immune cell", "monocyte": "Immune cell",
    "neutrophil": "Immune cell", "natural killer cell": "Immune cell",
    "mast cell": "Immune cell", "macrophage": "Immune cell",
    "cd8-positive, alpha-beta t cell": "Immune cell",
    "cd4-positive, alpha-beta t cell": "Immune cell",
    "plasmacytoid dendritic cell": "Immune cell",
    "conventional dendritic cell": "Immune cell",
    "innate lymphoid cell": "Immune cell", "kidney leukocyte cell": "Immune cell",
    "mhc class ii-positive cell": "Immune cell", "iga plasma cell": "Immune cell",
    "myeloid cell": "Immune cell", "plasma cell": "Immune cell",
    "t follicular helper cell": "Immune cell", "regulatory t cell": "Immune cell",
    "gamma-delta t cell": "Immune cell", "t-helper 17 cell": "Immune cell",
    "cycling macrophage": "Immune cell", "cycling t cell": "Immune cell",
    "kupffer cell": "Immune cell", "inflammatory macrophage": "Immune cell",
    "mature b cell": "Immune cell", "alpha-beta t cell": "Immune cell",
    "lymphocyte": "Immune cell",
    # Epithelial cell
    "thick ascending limb of loop of henle epithelial cell": "Epithelial cell",
    "kidney proximal tubule epithelial cell": "Epithelial cell",
    "renal principal cell": "Epithelial cell",
    "distal convoluted tubule epithelial cell": "Epithelial cell",
    "parietal epithelial cell": "Epithelial cell", "podocyte": "Epithelial cell",
    "renal beta-intercalated cell": "Epithelial cell",
    "renal alpha-intercalated cell": "Epithelial cell",
    "urothelial cell": "Epithelial cell", "nephron progenitor cell": "Epithelial cell",
    "nephron tubule epithelial cell": "Epithelial cell",
    "ureteric bud cell": "Epithelial cell",
    "kidney loop of henle epithelial cell": "Epithelial cell",
    "epithelial cell of large intestine": "Epithelial cell",
    "colonocyte": "Epithelial cell", "goblet cell": "Epithelial cell",
    "tuft cell": "Epithelial cell", "epithelial cell": "Epithelial cell",
    "pulmonary alveolar type 2 cell": "Epithelial cell",
    "mucus secreting cell": "Epithelial cell", "ciliated cell": "Epithelial cell",
    "club cell": "Epithelial cell", "basal cell": "Epithelial cell",
    "secretory cell": "Epithelial cell",
    "ciliated epithelial cell": "Epithelial cell",
    "pancreatic d cell": "Epithelial cell", "pancreatic ductal cell": "Epithelial cell",
    "pancreatic a cell": "Epithelial cell", "pancreatic pp cell": "Epithelial cell",
    "pancreatic acinar cell": "Epithelial cell",
    "type b pancreatic cell": "Epithelial cell",
    "pancreatic epsilon cell": "Epithelial cell",
    "pancreatic endocrine cell": "Epithelial cell",
    "hepatocyte": "Epithelial cell", "cholangiocyte": "Epithelial cell",
    "airway submucosal gland collecting duct epithelial cell": "Epithelial cell",
    "lung neuroendocrine cell": "Epithelial cell",
    "pulmonary ionocyte": "Epithelial cell",
    # Endothelial cell
    "endothelial cell": "Endothelial cell",
    "endothelial stalk cell": "Endothelial cell",
    "lymphatic endothelial cell": "Endothelial cell",
    "endothelial tip cell": "Endothelial cell",
    "endothelial cell of periportal hepatic sinusoid": "Endothelial cell",
    "endothelial cell of pericentral hepatic sinusoid": "Endothelial cell",
    # Stroma cell
    "fibroblast": "Stroma cell", "stromal cell": "Stroma cell",
    "myofibroblast": "Stroma cell", "pericyte": "Stroma cell",
    "smooth muscle cell": "Stroma cell", "pancreatic stellate cell": "Stroma cell",
    "hepatic stellate cell": "Stroma cell", "mesothelial cell": "Stroma cell",
    "mesenchymal cell": "Stroma cell",
    # Others
    "platelet": "Others", "erythroid lineage cell": "Others",
}

# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

preds = pd.read_csv(os.path.join(DATA_ROOT, "Fig2_zero_shot_all_cells_truth_pred_dataset.csv"))
preds["truth"] = preds["cell_type"].str.lower()
preds["pred"] = preds["predicted_cell_type"].str.lower()
preds["dataset_clean"] = preds["dataset"].str.replace(r"\.project$", "", regex=True)
preds["tissue"] = preds["dataset"].map(TISSUE_MAP)

summary = pd.read_csv(os.path.join(DATA_ROOT, "Fig2_zero_shot_all_metrics_summary.csv"))

print(f"Total cells: {len(preds):,}")
print(f"Datasets: {preds['dataset'].nunique()}")

# ============================================================
print("\n" + "=" * 60)
print("1. PER-DATASET METRICS")
print("=" * 60)

ds_rows = []
for ds in preds["dataset"].unique():
    sub = preds[preds["dataset"] == ds]
    acc = (sub["truth"] == sub["pred"]).mean()
    n_true_types = sub["truth"].nunique()
    n_pred_types = sub["pred"].nunique()
    tissue = TISSUE_MAP.get(ds, "Unknown")
    ds_clean = ds.replace(".project", "")

    # Get macro metrics from summary CSV
    s = summary[summary["dataset"] == ds]
    f1 = s["f1_macro"].values[0] if len(s) > 0 else np.nan
    prec = s["precision_macro"].values[0] if len(s) > 0 else np.nan
    rec = s["recall_macro"].values[0] if len(s) > 0 else np.nan

    ds_rows.append({
        "dataset": ds_clean, "tissue": tissue, "n_cells": len(sub),
        "n_cell_types": n_true_types, "accuracy": round(acc, 4),
        "f1_macro": f1, "precision_macro": prec, "recall_macro": rec,
    })

ds_df = pd.DataFrame(ds_rows).sort_values("f1_macro", ascending=False)
ds_df.to_csv(os.path.join(OUT_DIR, "zeroshot_per_dataset_metrics.csv"), index=False)

print(f"\n{'Dataset':<20s} {'Tissue':<10s} {'Cells':>8s} {'Types':>6s} {'Acc':>6s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s}")
print("-" * 75)
for _, r in ds_df.iterrows():
    print(f"{r['dataset']:<20s} {r['tissue']:<10s} {r['n_cells']:>8,d} {r['n_cell_types']:>6d} "
          f"{r['accuracy']:>6.2f} {r['f1_macro']:>6.2f} {r['precision_macro']:>6.2f} {r['recall_macro']:>6.2f}")

# ============================================================
print("\n" + "=" * 60)
print("2. TOP MISCLASSIFICATION PATTERNS PER DATASET")
print("=" * 60)

misclass_rows = []
for ds in sorted(preds["dataset"].unique()):
    sub = preds[preds["dataset"] == ds]
    wrong = sub[sub["truth"] != sub["pred"]]
    if len(wrong) == 0:
        continue

    confusion = wrong.groupby(["truth", "pred"]).size().reset_index(name="n_cells")
    true_counts = sub.groupby("truth").size().reset_index(name="true_total")
    confusion = confusion.merge(true_counts, on="truth")
    confusion["pct_of_true"] = (confusion["n_cells"] / confusion["true_total"] * 100).round(1)
    confusion = confusion.sort_values("n_cells", ascending=False).head(5)

    ds_clean = ds.replace(".project", "")
    tissue = TISSUE_MAP.get(ds, "Unknown")
    for _, r in confusion.iterrows():
        misclass_rows.append({
            "dataset": ds_clean, "tissue": tissue,
            "true_type": r["truth"], "predicted_type": r["pred"],
            "n_cells": r["n_cells"], "pct_of_true": r["pct_of_true"],
            "true_total": r["true_total"],
        })

misclass_df = pd.DataFrame(misclass_rows)
misclass_df.to_csv(os.path.join(OUT_DIR, "zeroshot_misclassification_patterns.csv"), index=False)

print(f"\nTop confusions per dataset:")
for ds in sorted(preds["dataset"].unique()):
    ds_clean = ds.replace(".project", "")
    ds_mis = misclass_df[misclass_df["dataset"] == ds_clean].head(3)
    if len(ds_mis) == 0:
        continue
    print(f"\n  {ds_clean} ({TISSUE_MAP.get(ds, '')}):")
    for _, r in ds_mis.iterrows():
        print(f"    {r['true_type'][:40]:<40s} → {r['predicted_type'][:30]:<30s} "
              f"{r['n_cells']:>6,d} ({r['pct_of_true']:.1f}%)")

# ============================================================
print("\n" + "=" * 60)
print("3. BROAD CELL LINEAGE METRICS (one-vs-rest)")
print("=" * 60)

preds["truth_broad"] = preds["truth"].map(BROAD_MAP).fillna("Others")
preds["pred_broad"] = preds["pred"].map(BROAD_MAP).fillna("Others")

lineage_rows = []
for cls in ["Epithelial cell", "Endothelial cell", "Immune cell", "Stroma cell", "Others"]:
    y_true = preds["truth_broad"] == cls
    y_pred = preds["pred_broad"] == cls
    TP = (y_true & y_pred).sum()
    FP = (~y_true & y_pred).sum()
    FN = (y_true & ~y_pred).sum()
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    lineage_rows.append({
        "lineage": cls, "F1": round(f1, 3), "Precision": round(prec, 3),
        "Recall": round(rec, 3), "support": int(y_true.sum()),
    })

lineage_df = pd.DataFrame(lineage_rows)
lineage_df.to_csv(os.path.join(OUT_DIR, "zeroshot_lineage_metrics.csv"), index=False)

print(f"\n{'Lineage':<20s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'Support':>8s}")
print("-" * 50)
for _, r in lineage_df.iterrows():
    print(f"{r['lineage']:<20s} {r['F1']:>6.3f} {r['Precision']:>6.3f} "
          f"{r['Recall']:>6.3f} {r['support']:>8,d}")

# ============================================================
print("\n" + "=" * 60)
print("4. SAMPLE SIZE BINS (one-vs-rest per fine type)")
print("=" * 60)

cell_types = sorted(preds["truth"].unique())
ct_rows = []
for ct in cell_types:
    y_true = preds["truth"] == ct
    y_pred = preds["pred"] == ct
    TP = (y_true & y_pred).sum()
    FP = (~y_true & y_pred).sum()
    FN = (y_true & ~y_pred).sum()
    prec = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    rec = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else np.nan
    ct_rows.append({"cell_type": ct, "F1": f1, "Precision": prec, "Recall": rec,
                    "support": int(y_true.sum())})

ct_df = pd.DataFrame(ct_rows)
ct_df["size_bin"] = pd.cut(ct_df["support"], bins=[0, 10, 100, float("inf")],
                           labels=["[0-10]", "[11-100]", "[>100]"], right=True)

ct_df.to_csv(os.path.join(OUT_DIR, "zeroshot_per_celltype_metrics.csv"), index=False)

size_agg = ct_df.groupby("size_bin").agg(
    F1=("F1", "mean"), Precision=("Precision", "mean"), Recall=("Recall", "mean"),
    n_types=("cell_type", "count")
).reset_index()
size_agg.to_csv(os.path.join(OUT_DIR, "zeroshot_size_bin_metrics.csv"), index=False)

print(f"\n{'Bin':<12s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'#Types':>8s}")
print("-" * 45)
for _, r in size_agg.iterrows():
    print(f"{r['size_bin']:<12s} {r['F1']:>6.3f} {r['Precision']:>6.3f} "
          f"{r['Recall']:>6.3f} {int(r['n_types']):>8d}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE. Outputs saved to:", OUT_DIR)
print("=" * 60)
