#!/usr/bin/env python3
"""
Compute the 6 embedding quality metrics for SCANPY and Seurat t-SNE coordinates,
matching the existing Fig3_topK_celltype_metrics.csv format.

Reads the upstream metrics for scLEAD, scTab, scFoundation, CellFM from the
read-only CSV in dung-code, drops the legacy 5000HVGs row, appends freshly
computed SCANPY (2,000 HVGs) and Seurat (2,000 HVGs) rows, and writes the
merged CSV to outputs/tables/Fig3_topK_celltype_metrics.csv. This path
is the one consumed by the plotting scripts and the supp-table generator;
dung-code is treated as read-only.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from collections import Counter

PROJECT_ROOT = "."
DATA_ROOT = os.path.join("./all_results", "1M_cells_70_cell_types")
# Read-only upstream file (source for the 4 non-HVG method rows).
UPSTREAM_TABLE = os.path.join(PROJECT_ROOT, "resources", "scLEAP", "results", "tables", "Fig3_topK_celltype_metrics.csv")
# Writable output consumed by src/plots/fig3_*.R and src/audit/gen_supp_tables.py.
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
OUTPUT_TABLE = os.path.join(OUTPUT_DIR, "Fig3_topK_celltype_metrics.csv")

TSNE_PATHS = {
    "SCANPY": os.path.join(DATA_ROOT, "HVG_TSNE_UMAP_scanpy_2000hvgs", "tsne_hvg.npy"),
    "Seurat": os.path.join(DATA_ROOT, "HVG_TSNE_UMAP_seurat_2000hvgs",
                           "runs", "n50_pcsN50_pca100_md0p75_tsne1_max1000000_p50_it2000_tpca40_s1_hvg2000_hvfmean.var.plot__1f0837eb4884",
                           "tsne_hvg.npy"),
}

TOP_K_VALUES = [10, 20, 30, 40, 50, 60, 70]


def load_labels():
    parquet_files = sorted(glob.glob(os.path.join(DATA_ROOT, "chunk_*.parquet")))
    dfs = [pd.read_parquet(f, columns=["cell_type"]) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)["cell_type"].values


def get_top_k_mask(labels, k):
    counts = Counter(labels)
    top_types = [t for t, _ in counts.most_common(k)]
    mask = np.isin(labels, top_types)
    return mask


def centroid_acc_macro(xy, labels):
    unique_labels = np.unique(labels)
    centroids = np.array([xy[labels == lab].mean(axis=0) for lab in unique_labels])
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    # For each cell, find nearest centroid
    # Vectorized: compute distances from each cell to all centroids
    # Process in chunks to avoid memory issues
    n = len(xy)
    correct_per_class = Counter()
    total_per_class = Counter()

    chunk_size = 50000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_xy = xy[start:end]
        chunk_labels = labels[start:end]
        dists = np.linalg.norm(chunk_xy[:, None, :] - centroids[None, :, :], axis=2)
        pred_idx = dists.argmin(axis=1)
        for i in range(end - start):
            true_lab = chunk_labels[i]
            total_per_class[true_lab] += 1
            if unique_labels[pred_idx[i]] == true_lab:
                correct_per_class[true_lab] += 1

    per_class_acc = [correct_per_class[lab] / total_per_class[lab] for lab in unique_labels]
    return np.mean(per_class_acc)


def centroid_ratio(xy, labels):
    unique_labels = np.unique(labels)
    centroids = np.array([xy[labels == lab].mean(axis=0) for lab in unique_labels])
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    n = len(xy)
    ratios = np.empty(n)
    chunk_size = 50000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_xy = xy[start:end]
        chunk_labels = labels[start:end]
        dists = np.linalg.norm(chunk_xy[:, None, :] - centroids[None, :, :], axis=2)
        for i in range(end - start):
            true_idx = label_to_idx[chunk_labels[i]]
            own_dist = dists[i, true_idx]
            other_dists = np.delete(dists[i], true_idx)
            nearest_other = other_dists.min()
            ratios[start + i] = own_dist / nearest_other if nearest_other > 0 else 0

    return ratios.mean()


def balanced_silhouette(xy, labels, max_per_class=2000, seed=1):
    unique_labels = np.unique(labels)
    from sklearn.metrics import silhouette_samples
    # Subsample per class to keep computation tractable
    rng = np.random.RandomState(seed)
    idx = []
    for lab in unique_labels:
        lab_idx = np.where(labels == lab)[0]
        if len(lab_idx) > max_per_class:
            lab_idx = rng.choice(lab_idx, max_per_class, replace=False)
        idx.append(lab_idx)
    idx = np.concatenate(idx)
    sub_xy = xy[idx]
    sub_labels = labels[idx]
    sil_samples = silhouette_samples(sub_xy, sub_labels, metric="euclidean")
    per_class_sil = [sil_samples[sub_labels == lab].mean() for lab in unique_labels]
    return np.mean(per_class_sil)


def calinski_harabasz(xy, labels):
    from sklearn.metrics import calinski_harabasz_score
    return calinski_harabasz_score(xy, labels)


def davies_bouldin(xy, labels):
    from sklearn.metrics import davies_bouldin_score
    return davies_bouldin_score(xy, labels)


def compute_metrics_for_method(method_name, tsne_path, all_labels):
    print(f"\nComputing metrics for {method_name}...")
    coords = np.load(tsne_path)
    assert coords.shape[0] == len(all_labels), f"Shape mismatch: {coords.shape[0]} vs {len(all_labels)}"

    rows = []
    for topk in TOP_K_VALUES:
        print(f"  Top K = {topk}...", end=" ", flush=True)
        mask = get_top_k_mask(all_labels, topk)
        xy = coords[mask]
        labels = all_labels[mask]

        ca = centroid_acc_macro(xy, labels)
        cr = centroid_ratio(xy, labels)
        sil = silhouette_score(xy, labels, metric="euclidean", sample_size=min(50000, len(xy)), random_state=1)
        bsil = balanced_silhouette(xy, labels)
        chi = calinski_harabasz(xy, labels)
        dbi = davies_bouldin(xy, labels)

        print(f"cent_acc={ca:.3f}, sil={sil:.3f}, dbi={dbi:.3f}")
        rows.append({
            "config": method_name,
            "cent_acc_macro": ca,
            "cent_ratio": cr,
            "sil": sil,
            "bsil": bsil,
            "chi_pm": chi,
            "dbi_pm": dbi,
            "topK": topk,
        })

    return pd.DataFrame(rows)


# Load labels
print("Loading labels...")
all_labels = load_labels()
print(f"  {len(all_labels):,} cells")

# Compute metrics for both methods
results = []
for method, path in TSNE_PATHS.items():
    df = compute_metrics_for_method(method, path, all_labels)
    results.append(df)

new_df = pd.concat(results, ignore_index=True)

# Load upstream CSV (read-only), remove legacy 5000HVGs row, add SCANPY and Seurat
existing = pd.read_csv(UPSTREAM_TABLE)
existing = existing[existing["config"] != "5000HVGs"]
updated = pd.concat([existing, new_df], ignore_index=True)

# Sort by config then topK
method_order = {"scLEAD": 0, "scTab": 1, "scFoundation": 2, "CellFM": 3, "SCANPY": 4, "Seurat": 5}
updated["sort_key"] = updated["config"].map(method_order)
updated = updated.sort_values(["sort_key", "topK"]).drop(columns=["sort_key"])
os.makedirs(OUTPUT_DIR, exist_ok=True)
updated.to_csv(OUTPUT_TABLE, index=False)
print(f"\nWrote {OUTPUT_TABLE}")
print(f"Methods in CSV: {updated['config'].unique().tolist()}")
