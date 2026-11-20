#!/usr/bin/env python3
"""Generate three supplementary LaTeX tables for the Results section:

1. supp:zeroshot-lineage   -- broad cell-lineage F1 / Precision / Recall (one-vs-rest)
2. supp:zeroshot-samplesize -- zero-shot per-cell-type metrics by sample-size bin
3. supp:embedding-metrics  -- t-SNE embedding-quality metrics across Top K = 10..70

Source data (all read-only in dung-code/):
- dung-code/resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv
  (per-cell truth + predicted for the 9 external zero-shot datasets)
- dung-code/resources/scLEAP/results/tables/Fig3_topK_celltype_metrics.csv
  (t-SNE embedding metrics across methods x Top K)

Outputs the LaTeX fragments to:
  outputs/tables/supp_zeroshot_lineage.tex
  outputs/tables/supp_zeroshot_samplesize.tex
  outputs/tables/supp_embedding_metrics.tex
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

ROOT = pathlib.Path(".")
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

ZS_CELLS = ROOT / "resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv"
TOPK_METRICS = ROOT / "outputs/tables/Fig3_topK_celltype_metrics.csv"
# ---------------------------------------------------------------------------
# Table 1: broad-lineage F1 / Precision / Recall (one-vs-rest, pooled over datasets)
# ---------------------------------------------------------------------------

METHOD_ORDER = ["scLEAD", "LangCell", "CellWhisperer"]


def table_lineage() -> str:
    """Per-method cell-type-group one-vs-rest F1/P/R with 95% bootstrap CIs.

    Values from `outputs/tables/zeroshot_group_bootstrap.csv` (1,000
    resamples, seed 1), matching Figure 2a's Cell Type Group sub-panel.
    """
    path = ROOT / "outputs/tables/zeroshot_group_bootstrap.csv"
    df = pd.read_csv(path)
    order = [
        "Pancreatic endocrine",
        "Pancreatic exocrine & stromal",
        "Renal tubule",
        "Renal parietal & urothelial",
        "Liver",
        "Airway & alveolar epithelial",
        "T / NK / innate lymphoid",
        "B / plasma",
        "Myeloid",
        "Stromal, endothelial & blood",
    ]
    df["group"] = pd.Categorical(df["group"], categories=order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "group"]).reset_index(drop=True)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{Zero-shot annotation performance by cell-type group for scLEAD and LangCell, "
         r"computed as one-vs-rest binary classification pooled across the ten external datasets. "
         r"Each cell type in the truth vocabulary is assigned to one of ten tissue- and lineage-informed groups. "
         r"$n$ is the number of cells with that true group label. "
         r"95\% confidence intervals come from 1{,}000 cell-level bootstrap resamples "
         r"(seed 1 shared across methods), matching Figure~2a's Cell Type Group sub-panel."
         r"}"),
        r"\label{supp:zeroshot-lineage}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l r l l l}",
        r"\toprule",
        (r"\textbf{Method} & \textbf{Cell Type Group} & \textbf{$n$} & "
         r"\textbf{Precision [95\% CI]} & \textbf{Recall [95\% CI]} & \textbf{F1 [95\% CI]} \\"),
        r"\midrule",
    ]
    current_method = None
    for _, r in df.iterrows():
        name = str(r["group"]).replace("&", r"\&")
        method_cell = r["method"] if current_method != r["method"] else ""
        if current_method is not None and current_method != r["method"]:
            lines.append(r"\midrule")
        current_method = r["method"]
        p_str = f"{r['Precision_observed']:.3f} [{r['Precision_ci_lo']:.3f}, {r['Precision_ci_hi']:.3f}]"
        r_str = f"{r['Recall_observed']:.3f} [{r['Recall_ci_lo']:.3f}, {r['Recall_ci_hi']:.3f}]"
        f_str = f"{r['F1_observed']:.3f} [{r['F1_ci_lo']:.3f}, {r['F1_ci_hi']:.3f}]"
        lines.append(
            f"{method_cell} & {name} & {int(r['n_cells']):,} & {p_str} & {r_str} & {f_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 2: zero-shot F1 by per-cell-type sample-size bin
# ---------------------------------------------------------------------------

def table_samplesize() -> str:
    """Per-method sample-size stratified one-vs-rest F1/P/R with 95% bootstrap CIs."""
    path = ROOT / "outputs/tables/zeroshot_samplesize_bootstrap.csv"
    df = pd.read_csv(path)
    label_map = {
        "[>2000]":     r"$>$2{,}000 cells",
        "[1001-2000]": r"1{,}001 to 2{,}000 cells",
        "[501-1000]":  r"501 to 1{,}000 cells",
        "[101-500]":   r"101 to 500 cells",
        "[51-100]":    r"51 to 100 cells",
        "[1-50]":      r"1 to 50 cells",
    }
    size_order = list(label_map.keys())
    df["size_bin"] = pd.Categorical(df["size_bin"], categories=size_order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "size_bin"]).reset_index(drop=True)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{Zero-shot performance for scLEAD and LangCell stratified by per-cell-type sample size, "
         r"pooled across the ten external datasets. "
         r"Each distinct cell type contributes one one-vs-rest F1, Precision, and Recall; "
         r"the bin-level value is the mean across those cell types. "
         r"95\% confidence intervals come from 1{,}000 cell-level bootstrap resamples "
         r"(seed 1 shared across methods), matching Figure~2a's Sample Size sub-panel. "
         r"$n_{ct}$ is the number of distinct cell types contributing to each bin."
         r"}"),
        r"\label{supp:zeroshot-samplesize}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l r l l l}",
        r"\toprule",
        (r"\textbf{Method} & \textbf{Sample size per cell type} & \textbf{$n_{ct}$} & "
         r"\textbf{Precision [95\% CI]} & \textbf{Recall [95\% CI]} & \textbf{F1 [95\% CI]} \\"),
        r"\midrule",
    ]
    current_method = None
    for _, r in df.iterrows():
        label = label_map[str(r["size_bin"])]
        if current_method is not None and current_method != r["method"]:
            lines.append(r"\midrule")
        method_cell = r["method"] if current_method != r["method"] else ""
        current_method = r["method"]
        if r["n_cell_types"] == 0:
            lines.append(f"{method_cell} & {label} & 0 & -- & -- & -- \\\\")
            continue
        def fmt(v, lo, hi):
            if pd.isna(v):
                return "--"
            lo_s = "--" if pd.isna(lo) else f"{lo:.3f}"
            hi_s = "--" if pd.isna(hi) else f"{hi:.3f}"
            return f"{v:.3f} [{lo_s}, {hi_s}]"
        p_str = fmt(r['Precision_observed'], r['Precision_ci_lo'], r['Precision_ci_hi'])
        r_str = fmt(r['Recall_observed'],    r['Recall_ci_lo'],    r['Recall_ci_hi'])
        f_str = fmt(r['F1_observed'],        r['F1_ci_lo'],        r['F1_ci_hi'])
        lines.append(
            f"{method_cell} & {label} & {int(r['n_cell_types'])} & {p_str} & {r_str} & {f_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: embedding-quality metrics across Top K
# ---------------------------------------------------------------------------

METHOD_LABEL = {
    "scLEAD": "scLEAD",
    "scTab": "scTab",
    "scFoundation": "scFoundation",
    "CellFM": "CellFM",
    "SCANPY": "SCANPY (2{,}000 HVGs)",
    "Seurat": "Seurat (2{,}000 HVGs)",
}
METRIC_LABEL = {
    "cent_acc_macro": "Centroid-based agreement",
    "cent_ratio": "Centroid compactness ratio",
    "sil": "Silhouette score",
    "bsil": "Balanced silhouette score",
    "chi_pm": "Calinski-Harabasz index",
    "dbi_pm": "Davies-Bouldin index",
}


def table_embedding() -> str:
    df = pd.read_csv(TOPK_METRICS)
    topks = sorted(df["topK"].unique())
    methods = ["scLEAD", "scTab", "scFoundation", "CellFM", "SCANPY", "Seurat"]
    metrics = ["cent_acc_macro", "cent_ratio", "sil", "bsil", "chi_pm", "dbi_pm"]

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{Embedding-quality metrics computed on the t-SNE projections of one "
         r"million held-out test cells (Figure~3b), across Top K most abundant cell types "
         r"(K = 10 to 70) for six representation methods. "
         r"Lower is better for the Davies-Bouldin index; higher is better for all other metrics. "
         r"SCANPY and Seurat use 2{,}000 highly variable genes each, matching the t-SNE "
         r"landscape in Figure~3a."
         r"}"),
        r"\label{supp:embedding-metrics}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l l " + " ".join(["r"] * len(topks)) + r"}",
        r"\toprule",
        r"\textbf{Metric} & \textbf{Method} & " + " & ".join([f"\\textbf{{K={k}}}" for k in topks]) + r" \\",
        r"\midrule",
    ]
    for m_i, metric in enumerate(metrics):
        for j, method in enumerate(methods):
            row = df[(df["config"] == method)].set_index("topK")[metric]
            cells = []
            for k in topks:
                v = row.get(k, np.nan)
                if metric == "chi_pm":
                    cells.append(f"{v:,.0f}" if pd.notna(v) else "--")
                else:
                    cells.append(f"{v:.3f}" if pd.notna(v) else "--")
            if j == 0:
                label = rf"\multirow{{{len(methods)}}}{{*}}{{{METRIC_LABEL[metric]}}}"
                lines.append(label + " & " + METHOD_LABEL[method] + " & " + " & ".join(cells) + r" \\")
            else:
                lines.append(" & " + METHOD_LABEL[method] + " & " + " & ".join(cells) + r" \\")
        if m_i < len(metrics) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_zeroshot_bootstrap() -> str:
    """Per-method per-dataset macro F1 / Precision / Recall with 95% CIs
    from `outputs/tables/zeroshot_per_dataset_bootstrap.csv`."""
    path = ROOT / "outputs/tables/zeroshot_per_dataset_bootstrap.csv"
    df = pd.read_csv(path)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df = df.sort_values(["method", "tissue", "F1_observed"], ascending=[True, True, False])

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        (r"\caption{Bootstrap 95\% confidence intervals for per-dataset zero-shot macro "
         r"F1, Precision, and Recall for scLEAD and LangCell on the ten external evaluation datasets. "
         r"CIs computed from 1{,}000 bootstrap resamples of cells within each dataset "
         r"(seed 1 shared across methods). ``Observed'' is the point estimate on the original assignment.}"),
        r"\label{supp:zeroshot-bootstrap}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l l r r l l l}",
        r"\toprule",
        (r"\textbf{Method} & \textbf{Dataset} & \textbf{Tissue} & \textbf{$n$ cells} & \textbf{\#\,types} & "
         r"\textbf{F1 [95\% CI]} & \textbf{Precision [95\% CI]} & \textbf{Recall [95\% CI]} \\"),
        r"\midrule",
    ]
    current_method = None
    for _, row in df.iterrows():
        if current_method is not None and current_method != row["method"]:
            lines.append(r"\midrule")
        method_cell = row["method"] if current_method != row["method"] else ""
        current_method = row["method"]
        f_str = f"{row['F1_observed']:.3f} [{row['F1_ci_lo']:.3f}, {row['F1_ci_hi']:.3f}]"
        p_str = f"{row['Precision_observed']:.3f} [{row['Precision_ci_lo']:.3f}, {row['Precision_ci_hi']:.3f}]"
        r_str = f"{row['Recall_observed']:.3f} [{row['Recall_ci_lo']:.3f}, {row['Recall_ci_hi']:.3f}]"
        lines.append(
            f"{method_cell} & {row['dataset']} & {row['tissue']} & {int(row['n_cells']):,} & "
            f"{int(row['n_cell_types'])} & {f_str} & {p_str} & {r_str} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    (OUT / "supp_zeroshot_lineage.tex").write_text(table_lineage())
    (OUT / "supp_zeroshot_samplesize.tex").write_text(table_samplesize())
    (OUT / "supp_zeroshot_bootstrap.tex").write_text(table_zeroshot_bootstrap())
    (OUT / "supp_embedding_metrics.tex").write_text(table_embedding())
    print(f"Wrote tables to {OUT}")


if __name__ == "__main__":
    main()
