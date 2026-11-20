#!/usr/bin/env python3
from __future__ import annotations

import os

# ---- hard limit BLAS threads (important when using multiprocessing) ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import random
import time
from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score

import scvi  # scvi-tools

# Ground truth label column
ct_key = "cell_type"


# -----------------------------
# Repro / deterministic sampling
# -----------------------------
def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_or_create_subsample_manifest(
    *, adata: sc.AnnData, dataset: str, subsample_dir: str, max_cells: int, seed: int
) -> List[str]:
    os.makedirs(subsample_dir, exist_ok=True)
    manifest_path = os.path.join(subsample_dir, f"{dataset}__max{max_cells}__seed{seed}.txt")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            cell_ids = [ln.strip() for ln in f if ln.strip()]
        present = [cid for cid in cell_ids if cid in adata.obs_names]
        if len(present) == 0:
            raise RuntimeError(f"Manifest exists but no cell IDs matched adata.obs_names: {manifest_path}")
        return present

    all_ids = adata.obs_names.to_list()
    if max_cells <= 0 or len(all_ids) <= max_cells:
        chosen = all_ids
    else:
        keys = [_sha1_hex(f"{seed}:{cid}") for cid in all_ids]
        order = np.argsort(np.array(keys, dtype=object))
        chosen = [all_ids[i] for i in order[:max_cells]]

    with open(manifest_path, "w") as f:
        for cid in chosen:
            f.write(cid + "\n")
    return chosen


# -----------------------------
# Tutorial-style counts handling
# -----------------------------
def _to_csr_float32(X):
    """Convert dense/sparse to CSR float32 (non-copy where possible)."""
    if sp.issparse(X):
        X = X.tocsr(copy=False)
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        return X
    arr = np.asarray(X)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return arr


def ensure_counts_layer_from_X(adata: sc.AnnData) -> None:
    """
    Tutorial pipeline wants raw counts in a layer (often `layers["counts"]`).
    We will:
      - If layers["counts"] exists: use it.
      - Else: set layers["counts"] = adata.X (as-is), converted to float32 CSR/dense.

    We DO NOT do normalize/log1p here (tutorial does NOT feed normalized/logged into scVI).
    """
    if "counts" in adata.layers:
        adata.layers["counts"] = _to_csr_float32(adata.layers["counts"])
    else:
        adata.layers["counts"] = _to_csr_float32(adata.X)


def sanitize_counts_layer_inplace(adata: sc.AnnData, *, clip_max: float) -> None:
    """
    Fixes HVG/binning failures from inf/nan, and prevents extreme values causing overflows.
    Operates only on layers["counts"].
    """
    X = adata.layers["counts"]
    if sp.issparse(X):
        data = X.data  # float32
        # Replace NaN/Inf
        bad = ~np.isfinite(data)
        if bad.any():
            data[bad] = 0.0
        # Ensure non-negative (counts-like)
        data[data < 0.0] = 0.0
        # Clip to prevent pathological values
        if clip_max is not None and clip_max > 0:
            np.clip(data, 0.0, float(clip_max), out=data)
        X.data = data
        X.eliminate_zeros()
        adata.layers["counts"] = X
    else:
        arr = np.asarray(X, dtype=np.float32)
        bad = ~np.isfinite(arr)
        if np.any(bad):
            arr[bad] = 0.0
        arr[arr < 0.0] = 0.0
        if clip_max is not None and clip_max > 0:
            np.clip(arr, 0.0, float(clip_max), out=arr)
        adata.layers["counts"] = arr


def disable_log1p_metadata(adata: sc.AnnData) -> None:
    """
    Some Scanpy functions will attempt expm1 if `adata.uns["log1p"]` exists, which can overflow
    on large values. For tutorial-style scVI, we want counts in a dedicated layer and we do not
    want Scanpy trying to "undo" log.
    """
    if "log1p" in adata.uns:
        adata.uns.pop("log1p", None)


# -----------------------------
# Main per-dataset run (tutorial pipeline)
# -----------------------------
def run_one_dataset(
    dataset: str,
    data_dir: str,
    result_dir: str,
    subsample_dir: str,
    max_cells: int,
    seed: int,
    max_epochs: int,
    resolution: float,
    n_top_genes: int,
    min_genes: int,
    min_cells: int,
    batch_key: Optional[str],
    use_gpu: bool,
    clip_max: float,
):
    start_total = time.time()
    out_dir = os.path.join(result_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)

    try:
        print(f"[RUN] {dataset}")
        set_random_seed(seed)

        dataset_path = os.path.join(data_dir, dataset, "test.h5ad")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Missing file: {dataset_path}")

        adata = sc.read(dataset_path)

        if ct_key not in adata.obs.columns:
            raise RuntimeError(f"Missing truth column '{ct_key}' in adata.obs for dataset={dataset}")

        # Deterministic subsample
        chosen_cells = get_or_create_subsample_manifest(
            adata=adata, dataset=dataset, subsample_dir=subsample_dir, max_cells=max_cells, seed=seed
        )
        if len(chosen_cells) < adata.n_obs:
            adata = adata[chosen_cells].copy()

        # Avoid Scanpy expm1 behavior caused by saved log1p metadata
        disable_log1p_metadata(adata)

        # Tutorial: use counts layer
        ensure_counts_layer_from_X(adata)
        sanitize_counts_layer_inplace(adata, clip_max=clip_max)

        # Tutorial-ish QC (optional but standard): mark mito + qc metrics on counts layer
        # (never required for scVI, but commonly done)
        try:
            upper = pd.Index(adata.var_names.astype(str)).str.upper()
            adata.var["mt"] = upper.str.startswith("MT-").to_numpy()
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], layer="counts", inplace=True)
        except Exception:
            # do not fail due to QC metadata
            pass

        # Filter on counts layer (tutorial: basic filtering is common)
        if min_genes and min_genes > 0:
            sc.pp.filter_cells(adata, min_genes=min_genes)
        if min_cells and min_cells > 0:
            sc.pp.filter_genes(adata, min_cells=min_cells)

        # Re-sanitize after filtering
        ensure_counts_layer_from_X(adata)
        sanitize_counts_layer_inplace(adata, clip_max=clip_max)

        # Tutorial: HVG selection on counts (Seurat v3)
        # If HVG fails for any reason, we log and proceed without subsetting (but scVI still runs).
        did_hvg = False
        if n_top_genes and n_top_genes > 0 and adata.n_vars > n_top_genes:
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    flavor="seurat_v3",
                    n_top_genes=n_top_genes,
                    layer="counts",
                    subset=True,
                )
                did_hvg = True
            except Exception as e:
                import traceback, sys

                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, "error.log"), "w") as f:
                    traceback.print_exc(file=f)

                print(f"[ERR] {dataset}: {e}", flush=True)
                raise

        # scVI setup (tutorial): point to counts layer
        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key=batch_key,
        )

        model = scvi.model.SCVI(adata)

        accelerator = "gpu" if (use_gpu and torch.cuda.is_available()) else "cpu"
        model.train(max_epochs=max_epochs, accelerator=accelerator)

        # Tutorial: latent -> neighbors -> Leiden
        adata.obsm["X_scVI"] = model.get_latent_representation()
        sc.pp.neighbors(adata, use_rep="X_scVI")
        sc.tl.leiden(adata, key_added="scvi_leiden", resolution=resolution)

        cluster_label = adata.obs["scvi_leiden"].astype(str)
        true_label = adata.obs[ct_key].astype(str)

        nmi = normalized_mutual_info_score(true_label, cluster_label)
        ari = adjusted_rand_score(true_label, cluster_label)
        ami = adjusted_mutual_info_score(true_label, cluster_label)

        # Save outputs
        cluster_df = pd.DataFrame(
            {"cell": adata.obs_names.astype(str), "cluster": cluster_label.values, "label": true_label.values}
        )
        cluster_df.to_csv(os.path.join(out_dir, "clusters.csv"), index=False)

        elapsed = time.time() - start_total
        with open(os.path.join(out_dir, "results.txt"), "w") as f:
            f.write(f"Tissue: {dataset}\n")
            f.write("Method: scVI+Leiden (tutorial-style)\n")
            f.write(f"NMI: {nmi}\n")
            f.write(f"ARI: {ari}\n")
            f.write(f"AMI: {ami}\n")
            f.write(f"Time_sec: {elapsed}\n")
            f.write(f"Max_cells: {max_cells}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Max_epochs: {max_epochs}\n")
            f.write(f"Resolution: {resolution}\n")
            f.write(f"N_top_genes: {n_top_genes}\n")
            f.write(f"Did_HVG: {did_hvg}\n")
            f.write(f"Min_genes: {min_genes}\n")
            f.write(f"Min_cells: {min_cells}\n")
            f.write(f"Batch_key: {batch_key if batch_key else ''}\n")
            f.write(f"Accelerator: {accelerator}\n")
            f.write(f"clip_max_counts: {clip_max}\n")
            f.write(f"n_obs_final: {adata.n_obs}\n")
            f.write(f"n_vars_final: {adata.n_vars}\n")

        print(f"[OK] {dataset} -> {out_dir}")

    except Exception as e:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "error_log.txt"), "a") as f:
            f.write(f"Error processing dataset {dataset}: {repr(e)}\n")
        print(f"[ERR] {dataset}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--subsample_dir", type=str, default=None)
    ap.add_argument("--max_cells", type=int, default=3000000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--max_epochs", type=int, default=100)

    # Tutorial-ish knobs
    ap.add_argument("--resolution", type=float, default=0.8)
    ap.add_argument("--n_top_genes", type=int, default=2000)
    ap.add_argument("--min_genes", type=int, default=200)
    ap.add_argument("--min_cells", type=int, default=3)
    ap.add_argument("--batch_key", type=str, default=None)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")

    # Safety for bad tissues (fix your current errors)
    ap.add_argument(
        "--clip_max",
        type=float,
        default=1e6,
        help="Clip counts layer values to [0, clip_max] after removing NaN/Inf (default: 1e6). "
             "Reduce if you still see overflows.",
    )

    args = ap.parse_args()

    subsample_dir = args.subsample_dir or os.path.join(args.result_dir, "_subsamples")
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(subsample_dir, exist_ok=True)

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]

    remain = [d for d in datasets if not os.path.exists(os.path.join(args.result_dir, d, "results.txt"))]
    for d in remain:
        run_one_dataset(
            dataset=d,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            subsample_dir=subsample_dir,
            max_cells=args.max_cells,
            seed=args.seed,
            max_epochs=args.max_epochs,
            resolution=args.resolution,
            n_top_genes=args.n_top_genes,
            min_genes=args.min_genes,
            min_cells=args.min_cells,
            batch_key=args.batch_key,
            use_gpu=(not args.cpu),
            clip_max=args.clip_max,
        )

    print("DONE")


if __name__ == "__main__":
    main()


"""
Install:
  pip install -U scvi-tools

Example:
  CUDA_VISIBLE_DEVICES=5 python run_scvi.py \
    --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
    --result_dir /data/share/dungp/single-cell/clustering-results/scvi \
    --max_cells 30000000 \
    --seed 1 \
    --max_epochs 100 \
    --n_top_genes 2000 \
    --resolution 0.8 \
    --clip_max 1000000
"""
