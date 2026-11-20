#!/usr/bin/env python3
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
import time
import random
import numpy as np
import pandas as pd
import scanpy as sc
import multiprocessing as mp
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    adjusted_mutual_info_score,
)

ct_key = "cell_type"


def set_random_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_or_create_subsample_manifest(
    *,
    adata: sc.AnnData,
    dataset: str,
    subsample_dir: str,
    max_cells: int,
    seed: int,
) -> list[str]:
    """
    Deterministically select cells by sorting cell_ids using SHA1(f"{seed}:{cell_id}")
    and taking the first max_cells. Save/read manifest so other methods can reuse the
    exact same cell set.
    """
    os.makedirs(subsample_dir, exist_ok=True)
    manifest_path = os.path.join(subsample_dir, f"{dataset}__max{max_cells}__seed{seed}.txt")

    # If manifest exists: reuse it (and intersect with current adata just in case)
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            cell_ids = [ln.strip() for ln in f if ln.strip()]
        present = [cid for cid in cell_ids if cid in adata.obs_names]
        if len(present) == 0:
            raise RuntimeError(f"Manifest exists but no cell IDs matched adata.obs_names: {manifest_path}")
        return present

    # If no subsampling needed, still write manifest for consistency across methods
    all_ids = adata.obs_names.to_list()
    if max_cells <= 0 or len(all_ids) <= max_cells:
        chosen = all_ids
    else:
        # deterministic rank by hash
        keys = [_sha1_hex(f"{seed}:{cid}") for cid in all_ids]
        order = np.argsort(np.array(keys, dtype=object))
        chosen = [all_ids[i] for i in order[:max_cells]]

    with open(manifest_path, "w") as f:
        for cid in chosen:
            f.write(cid + "\n")

    return chosen


def run_one_dataset(dataset: str, data_dir: str, result_dir: str, subsample_dir: str, max_cells: int, seed: int):
    try:
        print(dataset)
        set_random_seed(seed)

        dataset_path = os.path.join(data_dir, dataset, "test.h5ad")
        adata = sc.read(dataset_path)
        print(adata)

        # --------- deterministic subsampling (same cells across methods) ----------
        chosen_cells = get_or_create_subsample_manifest(
            adata=adata,
            dataset=dataset,
            subsample_dir=subsample_dir,
            max_cells=max_cells,
            seed=seed,
        )
        if len(chosen_cells) < adata.n_obs:
            adata = adata[chosen_cells].copy()
        # ------------------------------------------------------------------------

        start = time.time()

        # ---- keep clustering code unchanged from your script ----
        if adata.X.max() > 20:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        sc.tl.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, flavor="igraph", n_iterations=2)

        cluster_label = adata.obs["leiden"]
        true_label = adata.obs[ct_key]

        cluster_df = pd.DataFrame(
            {"cell": adata.obs_names.to_list(), "cluster": cluster_label, "label": true_label}
        )

        nmi = normalized_mutual_info_score(true_label, cluster_label)
        ari = adjusted_rand_score(true_label, cluster_label)

        silhouette = -1
        silhouette_cluster = -1
        ami = adjusted_mutual_info_score(true_label, cluster_label)

        path_to_save = os.path.join(result_dir, dataset)
        os.makedirs(path_to_save, exist_ok=True)

        cluster_df.to_csv(os.path.join(path_to_save, "cluster_df.csv"), index=False)

        with open(os.path.join(path_to_save, "results.txt"), "w") as f:
            f.write("Tissue: " + dataset + "\n")
            f.write("NMI: " + str(nmi) + "\n")
            f.write("ARI: " + str(ari) + "\n")
            f.write("AMI: " + str(ami) + "\n")
            f.write("silhouette: " + str(silhouette) + "\n")
            f.write("silhouette_true_celltype: " + str(silhouette) + "\n")
            f.write("silhouette_cluster: " + str(silhouette_cluster) + "\n")
            f.write("Time: " + str(time.time() - start) + "\n")

    except Exception as e:
        print(f"Error processing dataset {dataset}: {e}")
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "error_log.txt"), "a") as f:
            f.write(f"Error processing dataset {dataset}: {e}\n")


def _worker_init():
    # Re-assert thread limits inside each worker process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["BLAS_NUM_THREADS"] = "1"


def _run_star(args_tuple):
    return run_one_dataset(*args_tuple)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing <dataset>/test.h5ad")
    ap.add_argument("--result_dir", type=str, required=True, help="Output directory (per-dataset subfolders)")
    ap.add_argument("--subsample_dir", type=str, default=None, help="Where to store subsample manifests")
    ap.add_argument("--max_cells", type=int, default=3000000, help="Subsample to this many cells if larger")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", type=str, default=None, help="Run a single dataset name (folder)")

    # multiprocessing
    ap.add_argument("--parallel", action="store_true", help="Enable multiprocessing over datasets")
    ap.add_argument("--n_procs", type=int, default=4, help="Number of processes (default: 4)")
    ap.add_argument(
        "--mp_start",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method (default: spawn)",
    )
    ap.add_argument("--chunksize", type=int, default=1)

    args = ap.parse_args()
    subsample_dir = args.subsample_dir or os.path.join(args.result_dir, "_subsamples")
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(subsample_dir, exist_ok=True)

    if args.dataset is not None:
        datasets = [args.dataset]
    else:
        datasets = [d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))]

    remain = [d for d in datasets if not os.path.exists(os.path.join(args.result_dir, d, "results.txt"))]
    print(f"Datasets to process: {len(remain)} (parallel={args.parallel}, n_procs={args.n_procs})")

    job_args = [(d, args.data_dir, args.result_dir, subsample_dir, args.max_cells, args.seed) for d in remain]

    # sequential
    if (not args.parallel) or len(job_args) <= 1:
        for ja in job_args:
            _run_star(ja)
        print("DONE")
        return

    # multiprocessing
    try:
        mp.set_start_method(args.mp_start, force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context(args.mp_start)
    n_procs = max(1, min(args.n_procs, len(job_args)))
    print(f"[MP] start_method={args.mp_start} n_procs={n_procs} chunksize={args.chunksize}")

    with ctx.Pool(processes=n_procs, initializer=_worker_init) as pool:
        for _ in pool.imap_unordered(_run_star, job_args, chunksize=args.chunksize):
            pass

    print("DONE")


if __name__ == "__main__":
    main()

"""
# Sequential
python run_scanpy.py \
  --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --result_dir /data/share/dungp/single-cell/clustering-results/scanpy \
  --max_cells 300000 \
  --seed 1

# Parallel (default 4 processes)
python run_scanpy.py \
  --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --result_dir /data/share/dungp/single-cell/clustering-results/scanpy \
  --max_cells 300000 \
  --seed 1 \
  --parallel

# Parallel with explicit n_procs
python run_scanpy.py \
  --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --result_dir /data/share/dungp/single-cell/clustering-results/scanpy \
  --max_cells 300000 \
  --seed 1 \
  --parallel --n_procs 8 --mp_start spawn --chunksize 1
"""
