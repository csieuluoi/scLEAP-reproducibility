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
import random
import time
import numpy as np
import scanpy as sc
import multiprocessing as mp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from sklearn.ensemble import RandomForestClassifier
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


# ct_key = "cell_type_ontology_term_id"
# ct_key = "ontology_id"

random.seed(42)
np.random.seed(42)


def stratified_subsample(
    adata,
    label_key,
    n_cells=100000,
    random_state=42,
    min_per_strat_class=2,
):
    """
    Keep ALL singleton classes (count==1). Stratify only on classes with >= min_per_strat_class.
    Fix: handle categorical labels safely (remove unused categories + groupby(observed=True))
         to avoid empty groups that crash .sample().
    """
    rng = np.random.RandomState(random_state)

    obs_df = adata.obs.copy()
    obs_df["idx"] = np.arange(adata.n_obs)

    # Optional safety: handle missing labels explicitly
    # If you prefer to keep them, replace drop with fillna("unknown")
    if obs_df[label_key].isna().any():
        obs_df = obs_df[obs_df[label_key].notna()].copy()

    # If categorical, remove unused categories so groupby won't create empty groups
    s = obs_df[label_key]
    if pd.api.types.is_categorical_dtype(s):
        obs_df[label_key] = s.cat.remove_unused_categories()

    counts = obs_df[label_key].value_counts(dropna=False)

    # 1) singleton classes: always keep
    singleton_labels = counts[counts == 1].index
    singleton_df = obs_df[obs_df[label_key].isin(singleton_labels)]
    singleton_idx = singleton_df["idx"].to_numpy()

    # 2) stratifiable classes
    strat_labels = counts[counts >= min_per_strat_class].index
    strat_df = obs_df[obs_df[label_key].isin(strat_labels)].copy()

    if len(strat_df) == 0:
        if len(singleton_idx) == 0:
            raise ValueError("No samples found.")
        return adata[singleton_idx].copy()

    n_strat_classes = strat_df[label_key].nunique()
    min_required = len(singleton_idx) + n_strat_classes * min_per_strat_class
    if n_cells < min_required:
        raise ValueError(
            f"Requested n_cells={n_cells}, but need at least {min_required} to keep: "
            f"{len(singleton_idx)} singleton cells + {min_per_strat_class} per each of "
            f"{n_strat_classes} stratifiable classes."
        )

    # IMPORTANT: observed=True prevents empty groups from unused categories
    guaranteed_idx = (
        strat_df.groupby(label_key, sort=False, observed=True)["idx"]
        .apply(lambda s: s.sample(n=min_per_strat_class, random_state=random_state))
        .to_numpy()
    )

    already = set(singleton_idx.tolist()) | set(guaranteed_idx.tolist())
    remaining_budget = n_cells - len(already)

    remaining_pool = strat_df[~strat_df["idx"].isin(already)]
    take_n = min(remaining_budget, len(remaining_pool))

    if take_n > 0:
        extra_idx = remaining_pool.sample(n=take_n, random_state=random_state)["idx"].to_numpy()
    else:
        extra_idx = np.array([], dtype=int)

    final_idx = np.concatenate([singleton_idx, guaranteed_idx, extra_idx])
    rng.shuffle(final_idx)
    return adata[final_idx].copy()

def ensure_overlap_and_filter_test(adata_train, adata_test, label_key):
    """Keep only test cells whose labels appear in train labels."""
    train_labels = set(adata_train.obs[label_key].astype(str).values.tolist())
    test_labels = adata_test.obs[label_key].astype(str).values
    keep = np.array([lab in train_labels for lab in test_labels], dtype=bool)
    if keep.sum() == 0:
        return None, None
    adata_test = adata_test[keep].copy()
    adata_test.obs["idx2cal"] = True
    return adata_train, adata_test

def make_model(model_name: str, random_state: int, parallel: bool):
    """
    NO SVD.
    Use subsampling to manage scale.

    Multiprocessing fix:
      - For RF: n_jobs=1 when running multiprocessing (avoid oversubscription)
      - For LGBM: cap threads when in multiprocessing (avoid oversubscription)
    """
    model_name = model_name.lower()

    if model_name == "rf":
        return RandomForestClassifier(random_state=random_state, n_jobs=1 if parallel else None)

    if model_name == "lgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed. Install with:\n"
                "  conda install -c conda-forge lightgbm\n"
                "or\n"
                "  pip install lightgbm"
            )
        # "default params" + multiprocessing-safe thread cap
        # (num_threads is the one thing worth setting when you already parallelize by tissues)
        return LGBMClassifier(
            random_state=random_state,
            n_jobs=1 if parallel else None,
            # optional to reduce noise; remove if you want true defaults
            verbose=-1,
        )

    raise ValueError("model must be one of: rf, lgbm")

# def run_one_tissue(
#     tissue: str,
#     data_root: str,
#     save_root: str,
#     model_name: str,
#     use_subsample: bool,
#     subsample_n_cells: int,
#     min_per_strat_class: int,
#     parallel: bool,
#     random_state: int = 42,
# ):
#     start_time = time.time()
#     np.random.seed(random_state)

#     train_path = os.path.join(data_root, tissue, "train.h5ad")
#     test_path = os.path.join(data_root, tissue, "test.h5ad")
    
#     # save_root is the method folder; tissue saved directly under it
#     results_path = os.path.join(save_root, tissue)
#     os.makedirs(results_path, exist_ok=True)

#     if os.path.exists(os.path.join(results_path, "result.txt")):
#         print(f"[SKIP] {tissue}")
#         return

#     try:
#         print(f"[RUN] tissue={tissue} model={model_name.upper()} pid={os.getpid()}")
#         if tissue in ['blood', 'cerebral_cortex', 'lung', 'breast']:
#             ct_key = 'ontology_id'
#         else:
#             ct_key = 'cell_type_ontology_term_id'
#         data_train = sc.read_h5ad(train_path)
#         data_test = sc.read_h5ad(test_path)

#         original_n_cells = data_train.n_obs
#         was_subsampled = False

#         # subsample (train only)
#         if use_subsample and data_train.n_obs > subsample_n_cells:
#             data_train = stratified_subsample(
#                 data_train,
#                 label_key=ct_key,
#                 n_cells=subsample_n_cells,
#                 random_state=random_state,
#                 min_per_strat_class=min_per_strat_class,
#             )
#             was_subsampled = True

#         # normalize/log1p
#         sc.pp.normalize_total(data_train, target_sum=1e4)
#         sc.pp.log1p(data_train)
#         sc.pp.normalize_total(data_test, target_sum=1e4)
#         sc.pp.log1p(data_test)

#         # need >=2 classes in train
#         if len(data_train.obs[ct_key].unique()) < 2:
#             print(f"[SKIP] {tissue}: <2 classes")
#             return

#         # filter test to overlap labels
#         data_train, data_test = ensure_overlap_and_filter_test(data_train, data_test, label_key=ct_key)
#         if data_train is None:
#             print(f"[SKIP] {tissue}: no label overlap")
#             return

#         X_train = data_train.X
#         X_test = data_test.X
#         y_train = data_train.obs[ct_key].astype(str).values
#         y_test = data_test.obs[ct_key].astype(str).values

#         encoder = LabelEncoder()
#         y_train_num = encoder.fit_transform(y_train)

#         # diagnostics about singleton classes
#         vc = np.bincount(y_train_num)
#         min_cnt = int(vc.min()) if vc.size > 0 else 0
#         n_singletons = int((vc == 1).sum()) if vc.size > 0 else 0

#         model = make_model(model_name, random_state=random_state, parallel=parallel)

#         # Fit + predict
#         model.fit(X_train, y_train_num)
#         preds_num = model.predict(X_test)
#         preds = encoder.inverse_transform(preds_num)

#         # metrics
#         accuracy = accuracy_score(y_test, preds)
#         f1_macro = f1_score(y_test, preds, average="macro")
#         precision = precision_score(y_test, preds, average="macro", zero_division=0)
#         recall = recall_score(y_test, preds, average="macro", zero_division=0)
#         f1_weighted = f1_score(y_test, preds, average="weighted")
#         mcc = matthews_corrcoef(y_test, preds)
#         kappa = cohen_kappa_score(y_test, preds)
#         total_time = time.time() - start_time

#         with open(os.path.join(results_path, "result.txt"), "w") as f:
#             f.write(f"Tissue: {tissue}\n")
#             f.write(f"Model: {model_name.upper()}\n")
#             f.write(f"Train_min_class_count: {min_cnt}\n")
#             f.write(f"Train_num_singleton_classes: {n_singletons}\n")
#             if model_name.lower() == "rf" and parallel:
#                 f.write("RF_n_jobs: 1 (forced for multiprocessing)\n")
#             if model_name.lower() == "lgbm":
#                  f.write(f"LGBM_n_jobs: {1 if parallel else -1}\n")
#             f.write(f"Accuracy: {accuracy}\n")
#             f.write(f"F1 Macro: {f1_macro}\n")
#             f.write(f"F1 Weighted: {f1_weighted}\n")
#             f.write(f"Precision: {precision}\n")
#             f.write(f"Recall: {recall}\n")
#             f.write(f"Matthews Correlation Coefficient: {mcc}\n")
#             f.write(f"Cohen's Kappa: {kappa}\n")
#             f.write(f"Total Time: {total_time}\n")
#             f.write(f"Subsampled: {was_subsampled}\n")
#             f.write(f"Original_n_cells_train: {original_n_cells}\n")
#             f.write(f"Final_n_cells_train: {data_train.n_obs}\n")
#             if was_subsampled:
#                 f.write(f"Subsample_n_cells_target: {subsample_n_cells}\n")
#                 f.write(f"Subsample_min_per_strat_class: {min_per_strat_class}\n")

#         if was_subsampled:
#             with open(os.path.join(results_path, "subsampled.flag"), "w") as fflag:
#                 fflag.write("subsampled\n")

#         df_preds = data_test.obs[[ct_key]].copy()
#         df_preds['cell_type_ontology_term_id'] = df_preds.pop(ct_key)
#         df_preds["cell_id"] = data_test.obs_names
#         df_preds["predicted_cell_type"] = preds
#         df_preds["idx2cal"] = True
#         df_preds.to_csv(os.path.join(results_path, "predictions.csv"), index=False)

#         print(f"[OK] {tissue}: saved to {results_path}")

#     except Exception as e:
#         with open(os.path.join(results_path, "error.log"), "w") as f:
#             f.write(f"Error processing tissue {tissue}: {repr(e)}\n")
#         print(f"[ERR] {tissue}: {e}")

def run_one_tissue(
    tissue: str,
    data_root: str,
    save_root: str,
    model_name: str,
    use_subsample: bool,
    subsample_n_cells: int,
    min_per_strat_class: int,
    parallel: bool,
    random_state: int = 42,
):
    def log(msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)

    start_time = time.time()
    np.random.seed(random_state)

    train_path = os.path.join(data_root, tissue, "train.h5ad")
    test_path = os.path.join(data_root, tissue, "test.h5ad")

    results_path = os.path.join(save_root, tissue)
    os.makedirs(results_path, exist_ok=True)

    if os.path.exists(os.path.join(results_path, "result.txt")):
        log(f"[SKIP] {tissue}")
        return

    try:
        log(f"[RUN] tissue={tissue} model={model_name.upper()} pid={os.getpid()}")

        if tissue in ["blood", "cerebral_cortex", "lung", "breast"]:
            ct_key = "ontology_id"
        else:
            ct_key = "cell_type_ontology_term_id"

        # ---------------------------
        # LOAD DATA
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | loading train.h5ad")
        data_train = sc.read_h5ad(train_path)

        log(f"{tissue} | loading test.h5ad")
        data_test = sc.read_h5ad(test_path)

        log(
            f"{tissue} | loaded "
            f"train={data_train.n_obs}x{data_train.n_vars}, "
            f"test={data_test.n_obs}x{data_test.n_vars}, "
            f"time={time.time() - t0:.2f}s"
        )

        original_n_cells = data_train.n_obs
        was_subsampled = False

        # ---------------------------
        # SUBSAMPLE TRAIN
        # ---------------------------
        if use_subsample and data_train.n_obs > subsample_n_cells:
            log(
                f"{tissue} | SUBSAMPLE START "
                f"{data_train.n_obs} -> {subsample_n_cells}"
            )

            t0 = time.time()

            data_train = stratified_subsample(
                data_train,
                label_key=ct_key,
                n_cells=subsample_n_cells,
                random_state=random_state,
                min_per_strat_class=min_per_strat_class,
            )

            was_subsampled = True

            log(
                f"{tissue} | SUBSAMPLE DONE "
                f"final_train={data_train.n_obs}, "
                f"time={time.time() - t0:.2f}s"
            )
        else:
            log(
                f"{tissue} | SUBSAMPLE SKIP "
                f"use_subsample={use_subsample}, "
                f"train_n={data_train.n_obs}, "
                f"target={subsample_n_cells}"
            )

        # ---------------------------
        # NORMALIZE / LOG1P
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | NORMALIZE TRAIN START")
        sc.pp.normalize_total(data_train, target_sum=1e4)
        sc.pp.log1p(data_train)
        log(f"{tissue} | NORMALIZE TRAIN DONE time={time.time() - t0:.2f}s")

        t0 = time.time()
        log(f"{tissue} | NORMALIZE TEST START")
        sc.pp.normalize_total(data_test, target_sum=1e4)
        sc.pp.log1p(data_test)
        log(f"{tissue} | NORMALIZE TEST DONE time={time.time() - t0:.2f}s")

        # ---------------------------
        # CHECK LABELS
        # ---------------------------
        n_classes_before = data_train.obs[ct_key].nunique()
        log(f"{tissue} | train classes before overlap filtering = {n_classes_before}")

        if n_classes_before < 2:
            log(f"[SKIP] {tissue}: <2 classes")
            return

        # ---------------------------
        # FILTER TEST LABELS
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | LABEL OVERLAP FILTER START")

        data_train, data_test = ensure_overlap_and_filter_test(
            data_train,
            data_test,
            label_key=ct_key,
        )

        if data_train is None:
            log(f"[SKIP] {tissue}: no label overlap")
            return

        log(
            f"{tissue} | LABEL OVERLAP FILTER DONE "
            f"train={data_train.n_obs}, "
            f"test={data_test.n_obs}, "
            f"time={time.time() - t0:.2f}s"
        )

        # ---------------------------
        # PREPARE MATRICES
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | PREP MATRIX START")

        X_train = data_train.X
        X_test = data_test.X
        y_train = data_train.obs[ct_key].astype(str).values
        y_test = data_test.obs[ct_key].astype(str).values

        encoder = LabelEncoder()
        y_train_num = encoder.fit_transform(y_train)

        vc = np.bincount(y_train_num)
        min_cnt = int(vc.min()) if vc.size > 0 else 0
        n_singletons = int((vc == 1).sum()) if vc.size > 0 else 0

        log(
            f"{tissue} | PREP MATRIX DONE "
            f"X_train={X_train.shape}, "
            f"X_test={X_test.shape}, "
            f"classes={len(encoder.classes_)}, "
            f"min_class_count={min_cnt}, "
            f"singleton_classes={n_singletons}, "
            f"time={time.time() - t0:.2f}s"
        )

        # ---------------------------
        # BUILD MODEL
        # ---------------------------
        log(f"{tissue} | BUILD MODEL START")
        model = make_model(model_name, random_state=random_state, parallel=parallel)
        log(f"{tissue} | BUILD MODEL DONE model={model}")

        # ---------------------------
        # TRAIN
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | TRAINING START")

        model.fit(X_train, y_train_num)

        train_time = time.time() - t0
        log(f"{tissue} | TRAINING DONE time={train_time:.2f}s")

        # ---------------------------
        # PREDICT
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | PREDICTION START")

        preds_num = model.predict(X_test)
        preds = encoder.inverse_transform(preds_num)

        pred_time = time.time() - t0
        log(f"{tissue} | PREDICTION DONE time={pred_time:.2f}s")

        # ---------------------------
        # METRICS
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | METRICS START")

        accuracy = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average="macro")
        precision = precision_score(y_test, preds, average="macro", zero_division=0)
        recall = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, preds, average="weighted")
        mcc = matthews_corrcoef(y_test, preds)
        kappa = cohen_kappa_score(y_test, preds)

        log(f"{tissue} | METRICS DONE time={time.time() - t0:.2f}s")

        total_time = time.time() - start_time

        # ---------------------------
        # SAVE RESULT
        # ---------------------------
        t0 = time.time()
        log(f"{tissue} | SAVE RESULT START")

        with open(os.path.join(results_path, "result.txt"), "w") as f:
            f.write(f"Tissue: {tissue}\n")
            f.write(f"Model: {model_name.upper()}\n")
            f.write(f"Train_min_class_count: {min_cnt}\n")
            f.write(f"Train_num_singleton_classes: {n_singletons}\n")

            if model_name.lower() == "rf" and parallel:
                f.write("RF_n_jobs: 1 (forced for multiprocessing)\n")

            if model_name.lower() == "lgbm":
                f.write(f"LGBM_n_jobs: {1 if parallel else -1}\n")

            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"F1 Macro: {f1_macro}\n")
            f.write(f"F1 Weighted: {f1_weighted}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"Matthews Correlation Coefficient: {mcc}\n")
            f.write(f"Cohen's Kappa: {kappa}\n")
            f.write(f"Total Time: {total_time}\n")
            f.write(f"Training Time: {train_time}\n")
            f.write(f"Prediction Time: {pred_time}\n")
            f.write(f"Subsampled: {was_subsampled}\n")
            f.write(f"Original_n_cells_train: {original_n_cells}\n")
            f.write(f"Final_n_cells_train: {data_train.n_obs}\n")

            if was_subsampled:
                f.write(f"Subsample_n_cells_target: {subsample_n_cells}\n")
                f.write(f"Subsample_min_per_strat_class: {min_per_strat_class}\n")

        if was_subsampled:
            with open(os.path.join(results_path, "subsampled.flag"), "w") as fflag:
                fflag.write("subsampled\n")

        df_preds = data_test.obs[[ct_key]].copy()
        df_preds["cell_type_ontology_term_id"] = df_preds.pop(ct_key)
        df_preds["cell_id"] = data_test.obs_names
        df_preds["predicted_cell_type"] = preds
        df_preds["idx2cal"] = True
        df_preds.to_csv(os.path.join(results_path, "predictions.csv"), index=False)

        log(f"{tissue} | SAVE RESULT DONE time={time.time() - t0:.2f}s")
        log(f"[OK] {tissue}: saved to {results_path}, total_time={total_time:.2f}s")

    except Exception as e:
        with open(os.path.join(results_path, "error.log"), "w") as f:
            f.write(f"Error processing tissue {tissue}: {repr(e)}\n")

        log(f"[ERR] {tissue}: {repr(e)}")

def list_tissues_with_train_test(data_root: str):
    tissues, sizes = [], []
    for item in os.listdir(data_root):
        tissue_path = os.path.join(data_root, item)
        if not os.path.isdir(tissue_path):
            continue
        train_path = os.path.join(tissue_path, "train.h5ad")
        test_path = os.path.join(tissue_path, "test.h5ad")
        if os.path.exists(train_path) and os.path.exists(test_path):
            tissue_size = os.path.getsize(train_path) + os.path.getsize(test_path)
            tissues.append(item)
            sizes.append(tissue_size)
    tissues = [t for _, t in sorted(zip(sizes, tissues), reverse=False)]
    return tissues


def _worker_init():
    # Re-assert thread limits inside each worker process
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["BLAS_NUM_THREADS"] = "1"


def _run_one_tissue_star(args_tuple):
    return run_one_tissue(*args_tuple)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run RF / LGBM per tissue WITHOUT SVD. Use subsampling to handle large datasets."
        )
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "lgbm"])

    parser.add_argument("--tissue", type=str, default=None)

    # subsampling
    parser.add_argument("--use_subsample", action="store_true")
    parser.add_argument("--subsample_n_cells", type=int, default=500000)
    parser.add_argument("--min_per_strat_class", type=int, default=2)

    parser.add_argument("--random_state", type=int, default=42)

    # multiprocessing
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--n_procs", type=int, default=8)
    parser.add_argument("--mp_start", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    tissues = [args.tissue] if args.tissue is not None else list_tissues_with_train_test(args.data_root)

    with open('/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/new_cellxgene_download/meta-data/valid_tissues_full_data.txt', 'r') as f:
        valid_tissues_list = f.read().splitlines()

    tissues = [t for t in tissues if t in valid_tissues_list]

    if args.shuffle and args.tissue is None:
        rng = np.random.RandomState(args.random_state)
        rng.shuffle(tissues)

    valid_tissues = []
    for tissue in tissues:
        train_path = os.path.join(args.data_root, tissue, "train.h5ad")
        test_path = os.path.join(args.data_root, tissue, "test.h5ad")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            print(f"[SKIP] {tissue}: missing train/test.")
            continue
        valid_tissues.append(tissue)

    print(f"Total tissues to process: {len(valid_tissues)} (parallel={args.parallel})")

    job_args = [
        (
            tissue,
            args.data_root,
            args.save_root,
            args.model,
            args.use_subsample,
            args.subsample_n_cells,
            args.min_per_strat_class,
            args.parallel,
            args.random_state,
        )
        for tissue in valid_tissues
    ]

    if (not args.parallel) or len(job_args) <= 1:
        for ja in job_args:
            _run_one_tissue_star(ja)
        print("DONE")
        return

    try:
        mp.set_start_method(args.mp_start, force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context(args.mp_start)
    n_procs = max(1, min(args.n_procs, len(job_args)))
    print(f"[MP] start_method={args.mp_start} n_procs={n_procs} chunksize={args.chunksize}")

    with ctx.Pool(processes=n_procs, initializer=_worker_init) as pool:
        for _ in pool.imap_unordered(_run_one_tissue_star, job_args, chunksize=args.chunksize):
            pass

    print("DONE")


if __name__ == "__main__":
    main()

"""


python run_models_defaults_no_SVD.py \
  --data_root /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --save_root /data/share/dungp/single-cell/ct-annot-results/RF\
  --model rf \
  --use_subsample --subsample_n_cells 3000000 \
  --parallel --n_procs 6 --mp_start spawn


python run_models_defaults_no_SVD.py \
  --data_root /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --save_root /data/share/dungp/single-cell/ct-annot-results/GBM \
  --model lgbm \
  --use_subsample --subsample_n_cells 3000000 \
  --parallel --n_procs 6 --mp_start spawn

  


python run_models_defaults_no_SVD.py \
  --data_root /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --save_root /data/share/dungp/single-cell/ct-annot-results/RF\
  --model rf \
  --use_subsample --subsample_n_cells 3000000 \
  --tissue substantia_nigra_pars_compacta

python run_models_defaults_no_SVD.py \
  --data_root /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
  --save_root /data/share/dungp/single-cell/final_annot_results/LGBM \
  --model lgbm \
  --use_subsample --subsample_n_cells 300000 \
    --tissue blood

  
"""
