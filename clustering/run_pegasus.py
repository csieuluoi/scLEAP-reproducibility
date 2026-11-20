#!/usr/bin/env python3
"""
Run Pegasus clustering per tissue (Scanpy/Seurat-like benchmark layout).

Input layout (per tissue):
  --data_dir/<tissue>/test.h5ad

Outputs (per tissue):
  --result_dir/<tissue>/clusters.csv
  --result_dir/<tissue>/results.txt

Deterministic subsampling manifest (shared across methods):
  <result_dir>/_subsamples/<tissue>__max{max_cells}__seed{seed}.txt

Pegasus tutorial-style pipeline:
  log_norm -> highly_variable_features -> pca -> neighbors -> leiden

Optional:
  --cal_umap (computes UMAP using Pegasus)
  --cal_silhouette (requires UMAP; computes silhouette on UMAP coords)

Requirements:
  pip install pegasuspy
"""

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
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

import pegasus as pg
import multiprocessing as mp


# ----------------------------
# Deterministic subsampling
# ----------------------------
def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_or_create_subsample_manifest(
    *,
    cell_ids: List[str],
    dataset: str,
    subsample_dir: str,
    max_cells: int,
    seed: int,
) -> List[str]:
    """
    Deterministically select cells by sorting cell_ids using SHA1(f"{seed}:{cell_id}")
    and taking first max_cells. Save/read manifest so other methods can reuse exact cell set.
    """
    os.makedirs(subsample_dir, exist_ok=True)
    manifest_path = os.path.join(subsample_dir, f"{dataset}__max{max_cells}__seed{seed}.txt")

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            ids = [ln.strip() for ln in f if ln.strip()]
        if not ids:
            raise RuntimeError(f"Manifest exists but is empty: {manifest_path}")
        return ids

    if max_cells <= 0 or len(cell_ids) <= max_cells:
        chosen = list(cell_ids)
    else:
        keys = [_sha1_hex(f"{seed}:{cid}") for cid in cell_ids]
        order = np.argsort(np.array(keys, dtype=object))
        chosen = [cell_ids[i] for i in order[:max_cells]]

    with open(manifest_path, "w") as f:
        for cid in chosen:
            f.write(cid + "\n")

    return chosen


# ----------------------------
# Core runner
# ----------------------------
def run_one_dataset(
    dataset: str,
    *,
    data_dir: str,
    result_dir: str,
    subsample_dir: str,
    max_cells: int,
    seed: int,
    label_key: str,
    n_top_hvf: int,
    n_pcs: int,
    K: int,
    resolution: float,
    cal_umap: bool,
    cal_silhouette: bool,
    pegasus_n_jobs: int,
) -> None:
    out_dir = os.path.join(result_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.txt")

    if os.path.exists(results_path):
        print(f"[SKIP] {dataset}")
        return

    test_path = os.path.join(data_dir, dataset, "test.h5ad")
    start = time.time()

    try:
        print(f"[RUN] {dataset}")
        data = pg.read_input(test_path)

        # ---- deterministic subsampling (shared manifest across methods) ----
        cell_ids = list(data.obs_names)
        chosen = get_or_create_subsample_manifest(
            cell_ids=cell_ids,
            dataset=dataset,
            subsample_dir=subsample_dir,
            max_cells=max_cells,
            seed=seed,
        )
        chosen_present = [cid for cid in chosen if cid in data.obs_names]
        if not chosen_present:
            raise RuntimeError(f"No manifest cell IDs are present in {dataset} test.h5ad")

        if len(chosen_present) < data.shape[0]:
            data = data[chosen_present, :].copy()
        # ---------------------------------------------------------------

        if label_key not in data.obs.columns:
            raise KeyError(
                f"label_key '{label_key}' not found in obs. "
                f"Available: {list(data.obs.columns)[:50]} ..."
            )
        y_true = data.obs[label_key].astype(str).to_numpy()

        # Pegasus tutorial-like pipeline (defaults unless overridden)
        hvf_kwargs = {}
        pca_kwargs = {}
        neigh_kwargs = {}
        leiden_kwargs = {}
        umap_kwargs = {}

        if n_top_hvf != 2000:
            hvf_kwargs["n_top"] = int(min(n_top_hvf, data.shape[1]))
        if pegasus_n_jobs != 1:
            hvf_kwargs["n_jobs"] = int(pegasus_n_jobs)

        if n_pcs != 50:
            pca_kwargs["n_components"] = int(n_pcs)
        if pegasus_n_jobs != 1:
            pca_kwargs["n_jobs"] = int(pegasus_n_jobs)
        pca_kwargs["random_state"] = int(seed)

        if K != 100:
            neigh_kwargs["K"] = int(K)
        if pegasus_n_jobs != 1:
            neigh_kwargs["n_jobs"] = int(pegasus_n_jobs)

        if resolution != 1.3:
            leiden_kwargs["resolution"] = float(resolution)
        leiden_kwargs["random_state"] = int(seed)

        pg.log_norm(data)
        pg.highly_variable_features(data, **hvf_kwargs)
        pg.pca(data, **pca_kwargs)
        pg.neighbors(data, **neigh_kwargs)
        pg.leiden(data, **leiden_kwargs)

        y_pred = data.obs["leiden_labels"].astype(str).to_numpy()

        X_umap = None
        if cal_umap:
            if pegasus_n_jobs != 1:
                umap_kwargs["n_jobs"] = int(pegasus_n_jobs)
            umap_kwargs["random_state"] = int(seed)
            if K != 100:
                umap_kwargs["n_neighbors"] = int(K)
            pg.umap(data, **umap_kwargs)
            X_umap = np.asarray(data.obsm["X_umap"])

        nmi = normalized_mutual_info_score(y_true, y_pred)
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)

        sil_true = -1.0
        sil_cluster = -1.0
        if cal_silhouette:
            if X_umap is None:
                raise RuntimeError("--cal_silhouette requires --cal_umap.")
            sil_cluster = float(silhouette_score(X_umap, y_pred))
            try:
                sil_true = float(silhouette_score(X_umap, y_true))
            except Exception:
                sil_true = -1.0

        elapsed = time.time() - start

        pd.DataFrame({"cell": list(data.obs_names), "cluster": y_pred, "label": y_true}).to_csv(
            os.path.join(out_dir, "clusters.csv"), index=False
        )

        with open(results_path, "w") as f:
            f.write(f"Tissue: {dataset}\n")
            f.write("Method: Pegasus\n")
            f.write(f"n_cells: {data.shape[0]}\n")
            f.write(f"n_genes: {data.shape[1]}\n")
            f.write(f"max_cells: {max_cells}\n")
            f.write(f"seed: {seed}\n")

            if n_top_hvf != 2000:
                f.write(f"n_top_hvf: {n_top_hvf}\n")
            if n_pcs != 50:
                f.write(f"n_pcs: {n_pcs}\n")
            if K != 100:
                f.write(f"K: {K}\n")
            if resolution != 1.3:
                f.write(f"resolution: {resolution}\n")

            f.write(f"UMAP: {cal_umap}\n")
            f.write(f"NMI: {nmi}\n")
            f.write(f"ARI: {ari}\n")
            f.write(f"AMI: {ami}\n")
            f.write(f"silhouette_true: {sil_true}\n")
            f.write(f"silhouette_cluster: {sil_cluster}\n")
            f.write(f"Time: {elapsed}\n")

        print(f"[OK] {dataset}: saved -> {out_dir}")

    except Exception as e:
        with open(os.path.join(out_dir, "error.log"), "w") as f:
            f.write(f"Error processing dataset {dataset}: {repr(e)}\n")
        print(f"[ERR] {dataset}: {e}")


# ----------------------------
# Multiprocessing helpers (spawn-safe)
# ----------------------------
_GLOBAL_JOB_KWARGS: Dict[str, Any] = {}


def _pool_init(job_kwargs: Dict[str, Any]) -> None:
    # Re-assert thread limits inside each worker process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["BLAS_NUM_THREADS"] = "1"
    global _GLOBAL_JOB_KWARGS
    _GLOBAL_JOB_KWARGS = job_kwargs


def _worker_run(dataset: str) -> None:
    return run_one_dataset(dataset, **_GLOBAL_JOB_KWARGS)


def main():
    ap = argparse.ArgumentParser("Pegasus clustering benchmark with deterministic subsampling.")
    ap.add_argument("--data_dir", required=True, type=str, help="Directory containing <tissue>/test.h5ad")
    ap.add_argument("--result_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--subsample_dir", default=None, type=str, help="Manifest dir (default: <result_dir>/_subsamples)")
    ap.add_argument("--max_cells", default=3000000, type=int, help="Subsample size (<=0 means keep all)")
    ap.add_argument("--seed", default=1, type=int)
    ap.add_argument("--label_key", default="cell_type", type=str, help="obs column for ground truth labels")

    # Optional overrides (defaults follow Pegasus tutorial defaults)
    ap.add_argument("--n_top_hvf", default=2000, type=int)
    ap.add_argument("--n_pcs", default=50, type=int)
    ap.add_argument("--K", default=100, type=int, help="Pegasus neighbors K (default 100)")
    ap.add_argument("--resolution", default=1.3, type=float)

    ap.add_argument("--cal_umap", action="store_true", help="Compute UMAP (default off)")
    ap.add_argument("--cal_silhouette", action="store_true", help="Compute silhouette on UMAP (requires --cal_umap)")

    ap.add_argument("--dataset", default=None, type=str, help="Run a single tissue folder name")

    # multiprocessing
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--n_procs", type=int, default=4)
    ap.add_argument("--mp_start", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])
    ap.add_argument("--chunksize", type=int, default=1)

    # pegasus internal threading (keep 1 when using multiprocessing)
    ap.add_argument("--pegasus_n_jobs", type=int, default=1)

    args = ap.parse_args()

    set_random_seed(args.seed)
    os.makedirs(args.result_dir, exist_ok=True)
    subsample_dir = args.subsample_dir or os.path.join(args.result_dir, "_subsamples")
    os.makedirs(subsample_dir, exist_ok=True)

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = [
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
            and os.path.exists(os.path.join(args.data_dir, d, "test.h5ad"))
        ]

    datasets = sorted(datasets)
    remain = [d for d in datasets if not os.path.exists(os.path.join(args.result_dir, d, "results.txt"))]
    print(f"Total tissues to process: {len(remain)} (parallel={args.parallel})")

    job_kwargs = dict(
        data_dir=args.data_dir,
        result_dir=args.result_dir,
        subsample_dir=subsample_dir,
        max_cells=args.max_cells,
        seed=args.seed,
        label_key=args.label_key,
        n_top_hvf=args.n_top_hvf,
        n_pcs=args.n_pcs,
        K=args.K,
        resolution=args.resolution,
        cal_umap=args.cal_umap,
        cal_silhouette=args.cal_silhouette,
        pegasus_n_jobs=args.pegasus_n_jobs,
    )

    if (not args.parallel) or len(remain) <= 1:
        for d in remain:
            run_one_dataset(d, **job_kwargs)
        print("DONE")
        return

    try:
        mp.set_start_method(args.mp_start, force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context(args.mp_start)
    n_procs = max(1, min(args.n_procs, len(remain)))
    print(f"[MP] start_method={args.mp_start} n_procs={n_procs} chunksize={args.chunksize}")

    with ctx.Pool(
        processes=n_procs,
        initializer=_pool_init,
        initargs=(job_kwargs,),
    ) as pool:
        for _ in pool.imap_unordered(_worker_run, remain, chunksize=args.chunksize):
            pass

    print("DONE")


if __name__ == "__main__":
    main()

"""
Example:

python run_pegasus.py \
  --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --result_dir /data/share/dungp/single-cell/clustering-results/pegasus \
  --label_key cell_type \
  --max_cells 30000000 \
  --seed 1 \
  --parallel --n_procs 4 --mp_start spawn
"""
