#!/usr/bin/env python3
#%%
import os
import time
import argparse
import traceback

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import anndata as ad

# If you rely on relative imports from this repo, keep chdir optional via --workdir.
# os.chdir('/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/scFoundation')

from wrapper import SCFoundationLitModule
from dataset import H5ADDataset, LOG1PTransform, TotalSumNormalize
# from dataset_gpu import H5ADDatasetGPU


def run_tissue(
    path_train=None,
    path_test=None,
    path_to_save='.',
    tissue="perirenal_fat",
    seed=1,
    pool_type="all",       # 'mean' | 'max' | 'all'
    batch_size=32,
    max_epochs=1,
    devices=[0],           # PL devices (relative to CUDA_VISIBLE_DEVICES)
):
    """Train on train.h5ad, evaluate on test.h5ad, save metrics and pre.h5ad."""
    start = time.time()
    pl.seed_everything(seed, workers=True)

    if path_train is None:
        raise ValueError("path_train is None (should be set by caller).")
    if path_test is None:
        raise ValueError("path_test is None (should be set by caller).")

    label_key = "cell_type_ontology_term_id"
    transform = torch.nn.Sequential(LOG1PTransform(), TotalSumNormalize(target_sum=1e4))

    # CPU dataset (you had this active)
    train_ds = H5ADDataset(path_train, use_obs_column=label_key, transform=transform)
    test_ds  = H5ADDataset(path_test,  use_obs_column=label_key, transform=transform)

    train_label_dict = train_ds.label_dict
    train_int_to_label = {v: k for k, v in train_label_dict.items()}
    train_label_set = set(train_label_dict.keys())
    test_int_to_label  = {v: k for k, v in test_ds.label_dict.items()}

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        worker_init_fn=lambda i: np.random.seed(seed + i), generator=g
    )
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    wrapper = SCFoundationLitModule(ckpt_path='./model/models1.ckpt', num_classes=len(train_label_dict))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu',
        devices=devices,
        precision=16,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(wrapper, train_loader)

    wrapper.eval()
    wrapper.to('cuda')

    y_true_str, y_pred_str, emb_chunks = [], [], []

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for xb, yb in test_loader:
                xb = xb.to('cuda', non_blocking=True)
                yb = yb.to('cuda', non_blocking=True)

                logits, emb = wrapper.predict_with_embeddings(
                    xb,
                    pool_type=pool_type,
                    l2_normalize=True,
                )
                preds = torch.argmax(logits, dim=1)

                y_pred_str.extend([train_int_to_label[int(i)] for i in preds.detach().cpu().tolist()])
                y_true_str.extend([test_int_to_label[int(i)]  for i in yb.detach().cpu().tolist()])
                emb_chunks.append(emb.detach().cpu().numpy())

    E = np.concatenate(emb_chunks, axis=0)

    idx2cal = np.array([lbl in train_label_set for lbl in y_true_str], dtype=bool)

    os.makedirs(path_to_save, exist_ok=True)

    obs_df = pd.DataFrame({
        "true_label": y_true_str,
        "pred_label": y_pred_str,
        "idx2cal":    idx2cal
    })
    pre_path = os.path.join(path_to_save, "pre.h5ad")
    ad.AnnData(X=E, obs=obs_df).write_h5ad(pre_path)

    y_true_str = np.array(y_true_str)[idx2cal]
    y_pred_str = np.array(y_pred_str)[idx2cal]

    acc  = accuracy_score(y_true_str, y_pred_str)
    f1   = f1_score(y_true_str, y_pred_str, average='macro')
    rec  = recall_score(y_true_str, y_pred_str, average='macro', zero_division=0)
    prec = precision_score(y_true_str, y_pred_str, average='macro', zero_division=0)

    res_path = os.path.join(path_to_save, "result.txt")
    with open(res_path, "w") as f:
        f.write(f"Tissue:          {tissue}\n")
        f.write(f"Accuracy:        {acc:.6f}\n")
        f.write(f"F1 Macro:        {f1:.6f}\n")
        f.write(f"Recall:          {rec:.6f}\n")
        f.write(f"Precision:       {prec:.6f}\n")
        f.write(f"Total Time:      {time.time() - start:.2f}\n")

    del train_ds, test_ds, train_loader, test_loader, wrapper, trainer
    torch.cuda.empty_cache()

    return {
        "accuracy": acc, "f1_macro": f1, "recall_macro": rec, "precision_macro": prec,
        "pre_h5ad": pre_path, "result_path": res_path
    }


def list_tissues_from_data_root(data_root: str) -> list:
    tissues = []
    for item in os.listdir(data_root):
        tissue_path = os.path.join(data_root, item)
        if os.path.isdir(tissue_path):
            tissues.append(item)
    return sorted(tissues)


def load_tissues_from_file(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def filter_finished(tissues: list, save_root: str) -> list:
    finished = set()
    if os.path.isdir(save_root):
        for item in os.listdir(save_root):
            tissue_path = os.path.join(save_root, item)
            result_path = os.path.join(tissue_path, "result.txt")
            if os.path.isdir(tissue_path) and os.path.exists(result_path):
                finished.add(item)
    return [t for t in tissues if t not in finished]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, default=None, help="Optional: cd into this dir before running")
    parser.add_argument("--device", type=str, default=None, help="GPU id for CUDA_VISIBLE_DEVICES (e.g. 0,1,2).")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--save-root", type=str, required=True)

    # choose ONE of these sources
    parser.add_argument("--tissue", type=str, default=None, help="Run a single tissue name")
    parser.add_argument("--tissues-file", type=str, default=None, help="Text file with one tissue per line")
    parser.add_argument("--all", action="store_true", help="Run all tissues discovered in --data-root")

    parser.add_argument("--skip-finished", action="store_true", help="Skip tissues with save_root/<tissue>/result.txt")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pool-type", type=str, default="all", choices=["mean", "max", "all"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    args = parser.parse_args()

    if args.workdir:
        os.chdir(args.workdir)

    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    os.makedirs(args.save_root, exist_ok=True)

    # Decide tissues list
    if args.tissue:
        tissues = [args.tissue]
    elif args.tissues_file:
        tissues = load_tissues_from_file(args.tissues_file)
    elif args.all:
        tissues = list_tissues_from_data_root(args.data_root)
    else:
        raise SystemExit("Please specify one of: --tissue, --tissues-file, or --all")

    if args.skip_finished:
        tissues = filter_finished(tissues, args.save_root)

    print("Number of tissues to run:", len(tissues), flush=True)

    for tissue in tissues:
        print(f"Running tissue: {tissue}", flush=True)

        tissue_dir = os.path.join(args.data_root, tissue)
        path_train = os.path.join(tissue_dir, "train.h5ad")
        path_test  = os.path.join(tissue_dir, "test.h5ad")

        if not (os.path.exists(path_train) and os.path.exists(path_test)):
            print(f"Skip {tissue}: missing train/test h5ad", flush=True)
            continue

        path_to_save = os.path.join(args.save_root, tissue)
        os.makedirs(path_to_save, exist_ok=True)

        try:
            run_tissue(
                tissue=tissue,
                path_train=path_train,
                path_test=path_test,
                path_to_save=path_to_save,
                seed=args.seed,
                pool_type=args.pool_type,
                batch_size=args.batch_size,
                max_epochs=args.max_epochs,
                devices=[0],  # always 0 relative to CUDA_VISIBLE_DEVICES
            )
        except Exception as e:
            print(f"Error occurred for tissue {tissue}: {e}", flush=True)
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()


"""
python run_scFoundation_single_tissue.py \
    --data-root /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells_scFoundation/ \
    --save-root /data/share/dungp/single-cell/ct-annot-results/scFoundation/ \
    --tissue "adrenal_tissue" \
    --skip-finished \
    --device 7 \
    --batch-size 32 \
    --max-epochs 5

"""