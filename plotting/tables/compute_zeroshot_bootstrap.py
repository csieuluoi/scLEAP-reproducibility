#!/usr/bin/env python3
"""Compute bootstrap 95% confidence intervals for zero-shot macro F1, Precision,
and Recall for one or more methods on the nine external evaluation datasets.

For every method under `outputs/tables/method_predictions/*.csv`, the script
produces three summaries (each carries a leading `method` column):

    (1) per dataset         -> zeroshot_per_dataset_bootstrap.csv
    (2) per cell-type group -> zeroshot_group_bootstrap.csv   (10 groups)
    (3) per sample-size bin -> zeroshot_samplesize_bootstrap.csv

Reads (read-only):
    outputs/tables/method_predictions/<Method>.csv
        columns: dataset, cell_id, cell_type, predicted_cell_type  (CL lowercase)

All methods reuse `N_BOOT=1000` cell-level resamples and the same `SEED=1`, so
the bootstrap draws are shared across methods within each iteration and
differences reflect the methods, not the sampling.

Downstream consumers:
- src/plots/fig2_panel_b_zeroshot_metrics.R  (method-dodged error bars)
- src/audit/gen_supp_tables.py               (per-method supplementary tables)
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd


ROOT = pathlib.Path(".")
METHOD_DIR = ROOT / "outputs/tables/method_predictions"
OUT_DIR = ROOT / "outputs/tables"
OUT_CSV = OUT_DIR / "zeroshot_per_dataset_bootstrap.csv"
OUT_GROUP_CSV = OUT_DIR / "zeroshot_group_bootstrap.csv"
OUT_SIZE_CSV = OUT_DIR / "zeroshot_samplesize_bootstrap.csv"
OUT_FINE_CSV = OUT_DIR / "zeroshot_per_celltype_bootstrap.csv"

N_BOOT = 1000
SEED = 1

GROUP_ORDER = [
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

# Hand-curated mapping of every lowercase CL cell_type present in the 9
# zero-shot datasets to one of the 10 Figure 2a cell-type groups. Keep this in
# sync with the truth vocabulary; any unmapped type falls into the catch-all
# "Stromal, endothelial & blood" bucket and prints a warning at run time.
GROUP_MAP = {
    # 1. Pancreatic endocrine
    "pancreatic a cell": "Pancreatic endocrine",
    "type b pancreatic cell": "Pancreatic endocrine",
    "pancreatic d cell": "Pancreatic endocrine",
    "pancreatic pp cell": "Pancreatic endocrine",
    "pancreatic epsilon cell": "Pancreatic endocrine",
    # 2. Pancreatic exocrine & stromal
    "pancreatic acinar cell": "Pancreatic exocrine & stromal",
    "pancreatic ductal cell": "Pancreatic exocrine & stromal",
    "pancreatic stellate cell": "Pancreatic exocrine & stromal",
    # 3. Renal tubule
    "kidney proximal tubule epithelial cell": "Renal tubule",
    "distal convoluted tubule epithelial cell": "Renal tubule",
    "thick ascending limb of loop of henle epithelial cell": "Renal tubule",
    "renal principal cell": "Renal tubule",
    "renal alpha-intercalated cell": "Renal tubule",
    "renal beta-intercalated cell": "Renal tubule",
    # 4. Renal parietal & urothelial
    "podocyte": "Renal parietal & urothelial",
    "parietal epithelial cell": "Renal parietal & urothelial",
    "urothelial cell": "Renal parietal & urothelial",
    # 5. Liver
    "hepatocyte": "Liver",
    "cholangiocyte": "Liver",
    "kupffer cell": "Liver",
    "endothelial cell of pericentral hepatic sinusoid": "Liver",
    "endothelial cell of periportal hepatic sinusoid": "Liver",
    "hepatic stellate cell": "Liver",
    # 6. Airway & alveolar epithelial
    "basal cell": "Airway & alveolar epithelial",
    "ciliated cell": "Airway & alveolar epithelial",
    "club cell": "Airway & alveolar epithelial",
    "secretory cell": "Airway & alveolar epithelial",
    "goblet cell": "Airway & alveolar epithelial",
    "pulmonary alveolar type 2 cell": "Airway & alveolar epithelial",
    "pulmonary ionocyte": "Airway & alveolar epithelial",
    "lung neuroendocrine cell": "Airway & alveolar epithelial",
    "tuft cell": "Airway & alveolar epithelial",
    "mucus secreting cell": "Airway & alveolar epithelial",
    "epithelial cell": "Airway & alveolar epithelial",
    "ciliated epithelial cell": "Airway & alveolar epithelial",
    "airway submucosal gland collecting duct epithelial cell": "Airway & alveolar epithelial",
    # 7. T / NK / innate lymphoid
    "t cell": "T / NK / innate lymphoid",
    "alpha-beta t cell": "T / NK / innate lymphoid",
    "gamma-delta t cell": "T / NK / innate lymphoid",
    "cycling t cell": "T / NK / innate lymphoid",
    "natural killer cell": "T / NK / innate lymphoid",
    "lymphocyte": "T / NK / innate lymphoid",
    # 8. B / plasma
    "b cell": "B / plasma",
    "mature b cell": "B / plasma",
    "plasma cell": "B / plasma",
    # 9. Myeloid
    "monocyte": "Myeloid",
    "macrophage": "Myeloid",
    "neutrophil": "Myeloid",
    "cycling macrophage": "Myeloid",
    "inflammatory macrophage": "Myeloid",
    "conventional dendritic cell": "Myeloid",
    "plasmacytoid dendritic cell": "Myeloid",
    "mast cell": "Myeloid",
    "mhc class ii-positive cell": "Myeloid",
    "kidney leukocyte cell": "Myeloid",
    # 10. Stromal, endothelial & blood
    "fibroblast": "Stromal, endothelial & blood",
    "smooth muscle cell": "Stromal, endothelial & blood",
    "endothelial cell": "Stromal, endothelial & blood",
    "lymphatic endothelial cell": "Stromal, endothelial & blood",
    "platelet": "Stromal, endothelial & blood",
    "erythroid lineage cell": "Stromal, endothelial & blood",
    "mesothelial cell": "Stromal, endothelial & blood",
    "myofibroblast": "Stromal, endothelial & blood",
    "schwann cell": "Stromal, endothelial & blood",
}

SIZE_BINS = [
    ("[1-50]",       0,    50),
    ("[51-100]",     51,   100),
    ("[101-500]",    101,  500),
    ("[501-1000]",   501,  1000),
    ("[1001-2000]",  1001, 2000),
    ("[>2000]",      2001, 10**12),
]

# Some datasets annotate cells at child Cell Ontology terms (e.g. E-HCAD-9
# liver labels its T cells as "alpha-beta t cell" / "gamma-delta t cell")
# while others use the parent term ("t cell") for the same biological
# population. To remove this dataset-specific labelling-style penalty from the
# zero-shot scoring, every truth and prediction string below is collapsed to
# the parent term before any metric is computed. Applies symmetrically to all
# methods; cells whose truth label is not in this dict are unchanged.

# Cells whose truth label is an internal ancestor of other labels in the truth
# vocabulary are excluded from the pool. These are cells the dataset curator
# annotated at a parent CL term (e.g. generic "epithelial cell") even though a
# more specific CL child exists in the vocabulary, so a fine-grained
# prediction cannot be fairly scored against them. Keeping the panel at a
# single "antichain" level in the Cell Ontology avoids double-counting a
# lineage at both the parent and child level.
EXCLUDED_TRUTH_TYPES = {
    "epithelial cell",          # CL:0000066, parent of basal/ciliated/club/...
    "secretory cell",           # CL:0000151, parent of mucus secreting, goblet
    "mucus secreting cell",     # CL:0000319, parent of goblet cell
}
PARENT_COLLAPSE = {
    # T cell family -> "t cell"
    "alpha-beta t cell": "t cell",
    "gamma-delta t cell": "t cell",
    "cycling t cell": "t cell",
    # B cell family -> "b cell"
    "mature b cell": "b cell",
    # Macrophage family -> "macrophage"
    "kupffer cell": "macrophage",
    "inflammatory macrophage": "macrophage",
    "cycling macrophage": "macrophage",
    # Endothelial family -> "endothelial cell"
    "endothelial cell of pericentral hepatic sinusoid": "endothelial cell",
    "endothelial cell of periportal hepatic sinusoid": "endothelial cell",
    "lymphatic endothelial cell": "endothelial cell",
}

TISSUE_MAP = {
    "E-MTAB-9221": "Blood",
    "GSE185948": "Kidney",
    "E-HCAD-15": "Lung",
    "E-MTAB-5061": "Pancreas",
    "GSE86473": "Pancreas",
    "E-HCAD-9": "Liver",
    "GSE84133": "Pancreas",
    "E-CURD-114": "Lung",
    "GSE81608": "Pancreas",
    "GSE111976": "Endometrium",
}


def encode_labels(yt: np.ndarray, yp: np.ndarray):
    all_labels = np.unique(np.concatenate([yt, yp]))
    label_to_int = {lab: i for i, lab in enumerate(all_labels)}
    yt_int = np.fromiter((label_to_int[x] for x in yt), dtype=np.int64, count=len(yt))
    yp_int = np.fromiter((label_to_int[x] for x in yp), dtype=np.int64, count=len(yp))
    return yt_int, yp_int, len(all_labels)


def macro_prf_from_indices(yt_int: np.ndarray, yp_int: np.ndarray, C: int):
    cm = np.bincount(yt_int * C + yp_int, minlength=C * C).reshape(C, C)
    TP = np.diag(cm).astype(np.float64)
    colsum = cm.sum(axis=0).astype(np.float64)
    rowsum = cm.sum(axis=1).astype(np.float64)
    FP = colsum - TP
    FN = rowsum - TP
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(TP + FP > 0, TP / (TP + FP), 0.0)
        rec = np.where(TP + FN > 0, TP / (TP + FN), 0.0)
        denom = 2.0 * TP + FP + FN
        f1 = np.where(denom > 0, 2.0 * TP / denom, 0.0)
    return float(f1.mean()), float(prec.mean()), float(rec.mean())


def bootstrap_dataset(yt_int: np.ndarray, yp_int: np.ndarray, C: int,
                      n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(yt_int)
    f1s = np.empty(n_boot)
    precs = np.empty(n_boot)
    recs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        f, p, r = macro_prf_from_indices(yt_int[idx], yp_int[idx], C)
        f1s[i] = f
        precs[i] = p
        recs[i] = r
    return f1s, precs, recs


def per_class_prf(cm: np.ndarray):
    TP = np.diag(cm).astype(np.float64)
    colsum = cm.sum(axis=0).astype(np.float64)
    rowsum = cm.sum(axis=1).astype(np.float64)
    FP = colsum - TP
    FN = rowsum - TP
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(TP + FP > 0, TP / (TP + FP), np.nan)
        rec = np.where(TP + FN > 0, TP / (TP + FN), np.nan)
        denom = 2.0 * TP + FP + FN
        f1 = np.where(denom > 0, 2.0 * TP / denom, np.nan)
    return f1, prec, rec


def summarize(arr: np.ndarray, prefix: str) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_sd": float(np.std(arr, ddof=1)),
        f"{prefix}_ci_lo": float(np.percentile(arr, 2.5)),
        f"{prefix}_ci_hi": float(np.percentile(arr, 97.5)),
    }


def bootstrap_one_method(method: str, df: pd.DataFrame):
    """Run per-dataset + cell-type-group + sample-size bootstraps for one method.

    Returns three DataFrames each with a leading `method` column.
    """
    df = df.copy()
    df["cell_type"] = df["cell_type"].map(lambda x: PARENT_COLLAPSE.get(x, x))
    df["predicted_cell_type"] = df["predicted_cell_type"].map(
        lambda x: PARENT_COLLAPSE.get(x, x)
    )
    pre_n = len(df)
    df = df[~df["cell_type"].isin(EXCLUDED_TRUTH_TYPES)].reset_index(drop=True)
    dropped = pre_n - len(df)
    if dropped:
        print(f"  dropped {dropped:,} cells whose truth is a CL ancestor of "
              f"another label in the vocab", flush=True)
    # ---- Panel 1: per-dataset macro PRF ----
    per_dataset_rows = []
    for ds, sub in df.groupby("dataset", sort=True):
        yt = sub["cell_type"].astype(str).str.lower().to_numpy()
        yp = sub["predicted_cell_type"].astype(str).str.lower().to_numpy()
        yt_int, yp_int, C = encode_labels(yt, yp)
        f_obs, p_obs, r_obs = macro_prf_from_indices(yt_int, yp_int, C)
        f1s, precs, recs = bootstrap_dataset(yt_int, yp_int, C, N_BOOT, SEED)
        per_dataset_rows.append({
            "method": method,
            "dataset": ds,
            "tissue": TISSUE_MAP.get(ds, "?"),
            "n_cells": int(len(sub)),
            "n_cell_types": int(sub["cell_type"].nunique()),
            "F1_observed": f_obs,
            "Precision_observed": p_obs,
            "Recall_observed": r_obs,
            **summarize(f1s, "F1"),
            **summarize(precs, "Precision"),
            **summarize(recs, "Recall"),
        })
        print(
            f"  {method:10s} {ds:16s} {TISSUE_MAP.get(ds,'?'):9s} "
            f"F1={f_obs:.3f} [{per_dataset_rows[-1]['F1_ci_lo']:.3f},"
            f"{per_dataset_rows[-1]['F1_ci_hi']:.3f}]  n={len(sub):>7,}  C={C:>3d}",
            flush=True,
        )

    # ---- Panels 2 & 3: cell-type-group + sample-size one-vs-rest bootstraps ----
    yt_fine = df["cell_type"].astype(str).str.lower().to_numpy()
    yp_fine = df["predicted_cell_type"].astype(str).str.lower().to_numpy()

    unmapped_truth = sorted(set(yt_fine) - set(GROUP_MAP))
    if unmapped_truth:
        print(f"  WARNING unmapped truth cell types (assigned to last group): {unmapped_truth}",
              flush=True)
    catchall = GROUP_ORDER[-1]
    yt_g_str = np.array([GROUP_MAP.get(x, catchall) for x in yt_fine])
    yp_g_str = np.array([GROUP_MAP.get(x, catchall) for x in yp_fine])

    group_to_int = {c: i for i, c in enumerate(GROUP_ORDER)}
    yt_b = np.fromiter((group_to_int[x] for x in yt_g_str), dtype=np.int64,
                       count=len(yt_g_str))
    yp_b = np.fromiter((group_to_int[x] for x in yp_g_str), dtype=np.int64,
                       count=len(yp_g_str))
    Cb = len(GROUP_ORDER)

    all_fine = np.unique(np.concatenate([yt_fine, yp_fine]))
    fine_to_int = {c: i for i, c in enumerate(all_fine)}
    yt_f = np.fromiter((fine_to_int[x] for x in yt_fine), dtype=np.int64,
                       count=len(yt_fine))
    yp_f = np.fromiter((fine_to_int[x] for x in yp_fine), dtype=np.int64,
                       count=len(yp_fine))
    Cf = len(all_fine)

    truth_support = np.bincount(yt_f, minlength=Cf)
    ct_bin = np.full(Cf, -1, dtype=np.int64)
    for bi, (_, lo, hi) in enumerate(SIZE_BINS):
        ct_bin[(truth_support >= lo) & (truth_support <= hi)] = bi

    observed_group_f1, observed_group_p, observed_group_r = per_class_prf(
        np.bincount(yt_b * Cb + yp_b, minlength=Cb * Cb).reshape(Cb, Cb)
    )
    observed_fine_f1, observed_fine_p, observed_fine_r = per_class_prf(
        np.bincount(yt_f * Cf + yp_f, minlength=Cf * Cf).reshape(Cf, Cf)
    )

    def bin_means(per_ct_arr: np.ndarray):
        out = np.empty(len(SIZE_BINS))
        for bi in range(len(SIZE_BINS)):
            mask = ct_bin == bi
            out[bi] = np.nanmean(per_ct_arr[mask]) if mask.any() else np.nan
        return out

    observed_size_f1 = bin_means(observed_fine_f1)
    observed_size_p = bin_means(observed_fine_p)
    observed_size_r = bin_means(observed_fine_r)

    rng = np.random.default_rng(SEED)
    n = len(yt_f)
    group_f1s = np.empty((N_BOOT, Cb))
    group_ps = np.empty((N_BOOT, Cb))
    group_rs = np.empty((N_BOOT, Cb))
    size_f1s = np.empty((N_BOOT, len(SIZE_BINS)))
    size_ps = np.empty((N_BOOT, len(SIZE_BINS)))
    size_rs = np.empty((N_BOOT, len(SIZE_BINS)))
    fine_f1s = np.empty((N_BOOT, Cf))
    fine_ps = np.empty((N_BOOT, Cf))
    fine_rs = np.empty((N_BOOT, Cf))

    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        cm_b = np.bincount(yt_b[idx] * Cb + yp_b[idx], minlength=Cb * Cb).reshape(Cb, Cb)
        f_b, p_b, r_b = per_class_prf(cm_b)
        group_f1s[i] = f_b
        group_ps[i] = p_b
        group_rs[i] = r_b
        cm_f = np.bincount(yt_f[idx] * Cf + yp_f[idx], minlength=Cf * Cf).reshape(Cf, Cf)
        f_f, p_f, r_f = per_class_prf(cm_f)
        fine_f1s[i] = f_f
        fine_ps[i] = p_f
        fine_rs[i] = r_f
        size_f1s[i] = bin_means(f_f)
        size_ps[i] = bin_means(p_f)
        size_rs[i] = bin_means(r_f)

    def pct(arr, q):
        return np.nanpercentile(arr, q, axis=0)

    group_rows = []
    n_per_group = np.bincount(yt_b, minlength=Cb)
    for bi, cls in enumerate(GROUP_ORDER):
        group_rows.append({
            "method": method,
            "group": cls,
            "n_cells": int(n_per_group[bi]),
            "F1_observed": float(observed_group_f1[bi]),
            "F1_ci_lo": float(pct(group_f1s[:, bi], 2.5)),
            "F1_ci_hi": float(pct(group_f1s[:, bi], 97.5)),
            "Precision_observed": float(observed_group_p[bi]),
            "Precision_ci_lo": float(pct(group_ps[:, bi], 2.5)),
            "Precision_ci_hi": float(pct(group_ps[:, bi], 97.5)),
            "Recall_observed": float(observed_group_r[bi]),
            "Recall_ci_lo": float(pct(group_rs[:, bi], 2.5)),
            "Recall_ci_hi": float(pct(group_rs[:, bi], 97.5)),
        })

    size_rows = []
    for bi, (label, _, _) in enumerate(SIZE_BINS):
        size_rows.append({
            "method": method,
            "size_bin": label,
            "n_cell_types": int((ct_bin == bi).sum()),
            "F1_observed": float(observed_size_f1[bi]),
            "F1_ci_lo": float(pct(size_f1s[:, bi], 2.5)),
            "F1_ci_hi": float(pct(size_f1s[:, bi], 97.5)),
            "Precision_observed": float(observed_size_p[bi]),
            "Precision_ci_lo": float(pct(size_ps[:, bi], 2.5)),
            "Precision_ci_hi": float(pct(size_ps[:, bi], 97.5)),
            "Recall_observed": float(observed_size_r[bi]),
            "Recall_ci_lo": float(pct(size_rs[:, bi], 2.5)),
            "Recall_ci_hi": float(pct(size_rs[:, bi], 97.5)),
        })

    # Per-cell-type one-vs-rest summary (used for the top-15-most / least
    # abundant panels in Figure 2a). Cell types with zero truth support are
    # omitted because their one-vs-rest metrics are undefined.
    truth_support = np.bincount(yt_f, minlength=Cf)
    fine_rows = []
    for ci in range(Cf):
        if truth_support[ci] == 0:
            continue
        fine_rows.append({
            "method": method,
            "cell_type": all_fine[ci],
            "n_cells": int(truth_support[ci]),
            "F1_observed": float(observed_fine_f1[ci]) if not np.isnan(observed_fine_f1[ci]) else 0.0,
            "F1_ci_lo": float(pct(fine_f1s[:, ci], 2.5)),
            "F1_ci_hi": float(pct(fine_f1s[:, ci], 97.5)),
            "Precision_observed": float(observed_fine_p[ci]) if not np.isnan(observed_fine_p[ci]) else 0.0,
            "Precision_ci_lo": float(pct(fine_ps[:, ci], 2.5)),
            "Precision_ci_hi": float(pct(fine_ps[:, ci], 97.5)),
            "Recall_observed": float(observed_fine_r[ci]) if not np.isnan(observed_fine_r[ci]) else 0.0,
            "Recall_ci_lo": float(pct(fine_rs[:, ci], 2.5)),
            "Recall_ci_hi": float(pct(fine_rs[:, ci], 97.5)),
        })

    return (
        pd.DataFrame(per_dataset_rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(size_rows),
        pd.DataFrame(fine_rows),
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    method_files = sorted(METHOD_DIR.glob("*.csv"))
    if not method_files:
        raise SystemExit(f"no method CSVs under {METHOD_DIR}")

    t_total = time.time()
    per_ds, per_gr, per_sz, per_fine = [], [], [], []
    for path in method_files:
        method = path.stem
        print(f"\n=== {method} ({path.name}) ===", flush=True)
        df = pd.read_csv(path)
        print(f"  {len(df):,} cells across {df['dataset'].nunique()} datasets", flush=True)
        pd_rows, gr_rows, sz_rows, fine_rows = bootstrap_one_method(method, df)
        per_ds.append(pd_rows)
        per_gr.append(gr_rows)
        per_sz.append(sz_rows)
        per_fine.append(fine_rows)

    per_ds_df = pd.concat(per_ds, ignore_index=True).sort_values(
        ["method", "tissue", "F1_observed"], ascending=[True, True, False]
    )
    per_gr_df = pd.concat(per_gr, ignore_index=True)
    per_sz_df = pd.concat(per_sz, ignore_index=True)
    per_fine_df = pd.concat(per_fine, ignore_index=True)

    per_ds_df.to_csv(OUT_CSV, index=False)
    per_gr_df.to_csv(OUT_GROUP_CSV, index=False)
    per_sz_df.to_csv(OUT_SIZE_CSV, index=False)
    per_fine_df.to_csv(OUT_FINE_CSV, index=False)

    print(
        f"\nWrote {OUT_CSV}      ({len(per_ds_df)} rows)\n"
        f"Wrote {OUT_GROUP_CSV}    ({len(per_gr_df)} rows)\n"
        f"Wrote {OUT_SIZE_CSV} ({len(per_sz_df)} rows)\n"
        f"Wrote {OUT_FINE_CSV} ({len(per_fine_df)} rows)\n"
        f"{N_BOOT} bootstraps over {len(method_files)} methods, "
        f"total {time.time() - t_total:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
