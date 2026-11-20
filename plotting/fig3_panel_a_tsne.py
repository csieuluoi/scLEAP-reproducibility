#!/usr/bin/env python3
"""
Figure 3 Panel A: t-SNE landscape of 1M cells from 5 methods.
Outputs: figure/fig3_tsne_landscape.pdf

Reads t-SNE coordinates from .npy files and cell type labels from parquet chunks.
Subsamples to max_points for plotting speed. No cell type legend is drawn.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = "./"
DATA_ROOT = os.path.join("./all_results", "1M_cells_70_cell_types")
CONFIG_FILE = os.path.join(PROJECT_ROOT, "plotting", "config", "cluster_configs_all_representations.json")
OUT_FILE = os.path.join("figure", "fig3_tsne_landscape.pdf")

MAX_POINTS = 400_000  # subsample for plotting speed
SEED = 1
POINT_SIZE = 0.15
ALPHA = 0.5
TOP_K = 70  # top K most frequent cell types to show

# ============================================================
# FONT / SPACING CONTROLS
# ============================================================
TITLE_FONT_SIZE = 8
TITLE_PAD = 0
TITLE_Y = 0.98


HSPACE = 0.08
WSPACE = 0.05

TOP_MARGIN = 0.96
BOTTOM_MARGIN = 0.01
LEFT_MARGIN = 0.01
RIGHT_MARGIN = 0.99

# Method display names and order
METHOD_ORDER = [
    ("scLMCT_tsne30", "scLEAP"),
    ("scTab", "scTab"),
    ("scFoundation", "scFoundation"),
    ("CellFM", "CellFM"),
    ("scanpy", "SCANPY"),
    ("seurat", "Seurat"),
]

# ============================================================
# Load config to find npy paths
# ============================================================
with open(CONFIG_FILE) as f:
    config = json.load(f)

# Resolve paths: config uses remote paths, we replace with local base
def resolve_npy_path(method_key):
    """Find the tsne .npy file locally using the config structure."""
    info = config[method_key]
    # Configured paths are relative to the reproducibility repository root.
    remote_run_dir = os.path.abspath(info["run_dir"])

    # Get tsne filename from representations.tsne.path
    tsne_file = info.get("representations", {}).get("tsne", {}).get("path", "tsne_full.npy")

    # Strip remote prefix to get relative path from 1M_cells_70_cell_types
    prefix = "/data/share/dungp/single-cell/clustering-landscape/1M_cells_70_cell_types/"
    if remote_run_dir.startswith(prefix):
        rel_path = remote_run_dir[len(prefix):]
    else:
        rel_path = remote_run_dir

    local_path = os.path.join(DATA_ROOT, rel_path, tsne_file)
    if os.path.isfile(local_path):
        return local_path

    # Fallback: try both common names
    for fallback in ["tsne_full.npy", "tsne_hvg.npy"]:
        search = os.path.join(DATA_ROOT, rel_path, fallback)
        if os.path.isfile(search):
            return search

    # Fallback: recursive search under the rel_path directory
    search_dir = os.path.join(DATA_ROOT, rel_path)
    if os.path.isdir(search_dir):
        for root, dirs, files in os.walk(search_dir):
            if tsne_file in files:
                return os.path.join(root, tsne_file)

    raise FileNotFoundError(f"Cannot find t-SNE .npy for {method_key}. Tried: {local_path}")


# ============================================================
# Load labels from parquet chunks
# ============================================================
def load_labels():
    """Load cell_type labels from chunk_*.parquet files in row order."""
    parquet_files = sorted(glob.glob(os.path.join(DATA_ROOT, "chunk_*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No chunk_*.parquet in {DATA_ROOT}")
    dfs = [pd.read_parquet(f, columns=["cell_type"]) for f in parquet_files]
    labels = pd.concat(dfs, ignore_index=True)["cell_type"].values
    return labels


# ============================================================
# Stratified subsample
# ============================================================
def stratified_subsample(labels, max_points, seed):
    """Cap each cell type proportionally to max_points total."""
    rng = np.random.RandomState(seed)
    N = len(labels)
    if N <= max_points:
        return np.arange(N)

    unique_labels, counts = np.unique(labels, return_counts=True)
    ratio = max_points / N
    indices = []
    for lab, cnt in zip(unique_labels, counts):
        n_keep = max(1, int(round(cnt * ratio)))
        lab_idx = np.where(labels == lab)[0]
        if n_keep < len(lab_idx):
            chosen = rng.choice(lab_idx, n_keep, replace=False)
        else:
            chosen = lab_idx
        indices.append(chosen)
    return np.sort(np.concatenate(indices))


# ============================================================
# Build color map for top K cell types
# ============================================================
def build_colormap(labels, top_k):
    """Assign colors to the top_k most frequent cell types. Others get light gray.
    NPG palette cycled with modulo — repeated hues are acceptable here."""
    unique, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    top_types = unique[order[:top_k]]

    npg_palette = [
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
        "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
    ]
    color_map = {}
    for i, ct in enumerate(top_types):
        color_map[ct] = mcolors.to_rgba(npg_palette[i % len(npg_palette)])

    return color_map, top_types


# ============================================================
# Main
# ============================================================
print("Loading cell type labels from parquets...")
all_labels = load_labels()
N = len(all_labels)
print(f"  Total cells: {N:,}")

# Subsample
print(f"Subsampling to {MAX_POINTS:,} points...")
sub_idx = stratified_subsample(all_labels, MAX_POINTS, SEED)
sub_labels = all_labels[sub_idx]
print(f"  Subsampled: {len(sub_idx):,}")

# Build color map
color_map, top_types = build_colormap(all_labels, TOP_K)
top_mask = np.isin(all_labels, top_types)
print(f"  Top {TOP_K} cell types cover {top_mask.sum():,} cells")

# Assign colors
colors = np.array([color_map.get(lab, (0.85, 0.85, 0.85, 0.3)) for lab in sub_labels])

# Plot
fig, axes = plt.subplots(2, 3, figsize=(7.5, 4))
axes_flat = axes.flatten()

# Shuffle plot order for better visibility
rng = np.random.RandomState(SEED + 42)
plot_order = rng.permutation(len(sub_idx))

for i, (method_key, display_name) in enumerate(METHOD_ORDER):
    ax = axes_flat[i]
    print(f"Loading t-SNE for {display_name}...")
    npy_path = resolve_npy_path(method_key)
    coords = np.load(npy_path)
    if coords.shape[0] != N:
        raise ValueError(f"{display_name}: npy has {coords.shape[0]} rows, expected {N}")

    sub_coords = coords[sub_idx]

    ax.scatter(
        sub_coords[plot_order, 0],
        sub_coords[plot_order, 1],
        c=colors[plot_order],
        s=POINT_SIZE,
        alpha=ALPHA,
        edgecolors="none",
        rasterized=True,
    )

    ax.set_title(
        display_name,
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
        loc="center",
        pad=TITLE_PAD,
        y=TITLE_Y
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

plt.subplots_adjust(
    hspace=HSPACE,
    wspace=WSPACE,
    top=TOP_MARGIN,
    bottom=BOTTOM_MARGIN,
    left=LEFT_MARGIN,
    right=RIGHT_MARGIN
)

tmp_file = OUT_FILE + ".tmp.pdf"

print(f"Saving to {OUT_FILE}...")
fig.savefig(tmp_file, dpi=300, bbox_inches="tight")
plt.close(fig)

import subprocess

subprocess.run(
    ["pdfcrop", tmp_file, OUT_FILE],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

os.remove(tmp_file)

print(f"Done. Saved: {OUT_FILE}")