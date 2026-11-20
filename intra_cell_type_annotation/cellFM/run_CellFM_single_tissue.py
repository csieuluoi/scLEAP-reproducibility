#%%
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=5, help='GPU device id to use')
parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing Pool.")
parser.add_argument("--max-workers", type=int, default=4, help="Number of parallel workers if --parallel is set.")

# NEW: single-tissue mode
parser.add_argument(
    "--tissue",
    type=str,
    default=None,
    help="Run only this tissue (folder name under data_root). If set, ignores --parallel and --max-workers."
)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

print("Current working directory:", os.getcwd())

import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from layers.utils import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
import json
import multiprocessing as mp
import traceback

import warnings
warnings.filterwarnings("ignore")

from model import Cell_FM

ct_key = 'cell_type_ontology_term_id'


def load_train_test(tissue_root):
    train_h5ad = os.path.join(tissue_root, "train.h5ad")
    test_h5ad  = os.path.join(tissue_root, "test.h5ad")

    cell_type_mapping = {}
    # ---- TRAIN ----
    adata_train = read_h5ad(train_h5ad)
    for ct, i in enumerate(adata_train.obs[ct_key].unique()):
        cell_type_mapping[i] = ct
    adata_train.obs['celltype'] = adata_train.obs[ct_key].map(cell_type_mapping)
    adata_train.obs["celltype_id"] = adata_train.obs[ct_key].map(cell_type_mapping)
    adata_train.obs["feat"] = adata_train.obs[cfg.feature_col].cat.codes.values
    cfg.num_cls = len(adata_train.obs["feat"].unique())   # original behavior
    adata_train.obs["batch_id"] = 0
    adata_train.obs["train"] = 0
    adata_train.var_names = adata_train.var['feature']

    train_loader = build_dataset(
        SCrna(adata_train, mode="train"),
        prep=Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio),
        batch_size=cfg.use_bs,
        pad_zero=cfg.pad_zero,
        drop=True,
        shuffle=True,
    )

    # ---- TEST ----
    adata_test = read_h5ad(test_h5ad)
    adata_test = adata_test[adata_test.obs[ct_key].isin(adata_train.obs[ct_key].unique())].copy()
    adata_test.obs['celltype'] = adata_test.obs[ct_key]
    adata_test.obs["celltype_id"] = adata_test.obs[ct_key].map(cell_type_mapping)

    adata_test.obs["feat"] = adata_test.obs[cfg.feature_col].cat.codes.values
    adata_test.obs["batch_id"] = 0
    adata_test.obs["train"] = 2
    adata_test.var_names = adata_test.var['feature']

    test_loader = build_dataset(
        SCrna(adata_test, mode="test"),
        prep=Prepare(cfg.nonz_len, pad=0, mask_ratio=cfg.mask_ratio),
        batch_size=cfg.use_bs,
        drop=False,
        shuffle=False,
    )

    return train_loader, test_loader, adata_test.obs, cell_type_mapping


class Finetune_Cell_FM(nn.Module):
    def __init__(self, cfg):
        super(Finetune_Cell_FM, self).__init__()
        self.cfg = cfg
        self.num_cls = cfg.num_cls
        self.extractor = Cell_FM(27855, self.cfg, ckpt_path=self.cfg.ckpt_path, device=self.cfg.device)
        self.cls = nn.Sequential(
            nn.Linear(self.cfg.enc_dims, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_cls)
        )

    def forward(self, raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx):
        mask_loss, cls_token = self.extractor(
            raw_nzdata, dw_nzdata, ST_feat, nonz_gene, mask_gene, zero_idx
        )
        cls = self.cls(cls_token)
        return cls, mask_loss, cls_token


def save_emb(emb, pred, label, save_path, mode, cell_ids=None):
    if len(emb) > 1:
        emb = np.vstack([np.atleast_2d(x) for x in emb])
        pred = np.concatenate([np.ravel(x) for x in pred])
        label = np.concatenate([np.ravel(x) for x in label])
    else:
        emb = np.array(emb[0])
        pred = np.ravel(pred[0])
        label = np.ravel(label[0])

    adata = sc.AnnData(
        X=emb,
        obs=pd.DataFrame({"pred": pred, "label": label})
    )
    if cell_ids is not None:
        adata.obs_names = cell_ids
    os.makedirs(save_path, exist_ok=True)
    sc.write(f"{save_path}/{mode}.h5ad", adata)


cfg = Config_80M()
cfg.ecs_threshold = 0.8
cfg.ecs = True
cfg.add_zero = True
cfg.pad_zero = True
cfg.use_bs = 32
cfg.mask_ratio = 0.5
cfg.feature_col = ct_key
cfg.ckpt_path = "/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/cellFM/para80m/CellFM_80M_weight.ckpt"
cfg.device = "cuda:0"
cfg.epoch = 10
cfg.num_cls = 1


def run_dataset(save_root, tissue_name):
    all_time = {}
    start = time.time()
    cfg.dataset = tissue_name

    PT_PATH = f"{save_root}/{cfg.dataset}"
    MODEL_PATH = f"{save_root}/{cfg.dataset}/model_checkpoint/"

    os.makedirs(PT_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)

    tissue_root = f"/data/share/pbya/sc-dung-data/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells/{cfg.dataset}"

    train_loader, test_loader, test_obs, cell_type_mapping = load_train_test(tissue_root)
    process_time = time.time() - start
    print(f"Data loading time: {process_time:.2f} seconds")
    all_time['data processing'] = process_time
    with open(os.path.join(PT_PATH, "timing.json"), "w") as f:
        json.dump(all_time, f, indent=4)

    start_time = time.time()

    net = Finetune_Cell_FM(cfg)
    for name, param in net.named_parameters():
        param.requires_grad = ("cls." in name) or ("encoder" in name)

    print("Trainable parameters:")
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
    print(net)
    print('number of trainable parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))
    print("Total number of parameters:", sum(p.numel() for p in net.parameters()))
    net = net.to(cfg.device)
    net.extractor.load_model(weight=True, moment=False)

    optimizer = AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-5)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    criterion_cls = nn.CrossEntropyLoss()

    loading_model_time = time.time() - start_time
    print(f"Model loading time: {loading_model_time:.2f} seconds")
    all_time['model loading'] = loading_model_time
    with open(os.path.join(PT_PATH, "timing.json"), "w") as f:
        json.dump(all_time, f, indent=4)

    start_time = time.time()

    for epoch in range(cfg.epoch):
        train_emb, train_pred, train_label = [], [], []
        net.train()
        print("training...")
        running_loss = 0.0
        running_acc = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epoch}")

        for step, batch in enumerate(progress):
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            celltype_label = batch['celltype_label'].long().to(cfg.device)
            batch_id = batch['batch_id'].to(cfg.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                cls, mask_loss, cls_token = net(
                    raw_nzdata=raw_nzdata,
                    dw_nzdata=dw_nzdata,
                    ST_feat=ST_feat,
                    nonz_gene=nonz_gene,
                    mask_gene=mask_gene,
                    zero_idx=zero_idx
                )
                cls_loss = criterion_cls(cls, celltype_label)
                loss = mask_loss + cls_loss

            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            pred = cls.argmax(1)
            accuracy = (pred == celltype_label).sum().item() / len(batch_id)

            running_loss += loss.item()
            running_acc += accuracy

            avg_loss = running_loss / (step + 1)
            avg_acc = running_acc / (step + 1)
            progress.set_postfix(loss=avg_loss, acc=avg_acc)

            train_emb.append(cls_token.detach().cpu().numpy().tolist())
            train_label.append(celltype_label.detach().cpu().numpy().tolist())
            train_pred.append(pred.detach().cpu().numpy().tolist())

        scheduler.step()
        print(f"Epoch {epoch+1} complete,avg loss: {avg_loss:.6f}")

    torch.save(net.state_dict(), f"{MODEL_PATH}/checkpoint.pth")

    finetune_time = time.time() - start_time
    print(f"Fine-tuning time: {finetune_time:.2f} seconds")
    all_time['fine-tuning'] = finetune_time
    with open(os.path.join(PT_PATH, "timing.json"), "w") as f:
        json.dump(all_time, f, indent=4)

    start_time = time.time()

    net.eval()
    print("evaluating...")
    test_emb, test_pred, test_label = [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            raw_nzdata = batch['raw_nzdata'].to(cfg.device)
            dw_nzdata = batch['dw_nzdata'].to(cfg.device)
            ST_feat = batch['ST_feat'].to(cfg.device)
            nonz_gene = batch['nonz_gene'].to(cfg.device)
            mask_gene = batch['mask_gene'].to(cfg.device)
            zero_idx = batch['zero_idx'].to(cfg.device)
            celltype_label = batch['celltype_label'].long().to(cfg.device)

            cls, mask_loss, cls_token = net(
                raw_nzdata=raw_nzdata,
                dw_nzdata=dw_nzdata,
                ST_feat=ST_feat,
                nonz_gene=nonz_gene,
                mask_gene=mask_gene,
                zero_idx=zero_idx
            )
            pred = cls.argmax(1)

            test_emb.append(cls_token.detach().cpu().numpy().tolist())
            test_label.append(celltype_label.detach().cpu().numpy().tolist())
            test_pred.append(pred.detach().cpu().numpy().tolist())

    evaluation_time = time.time() - start_time
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    all_time['evaluation'] = evaluation_time
    with open(os.path.join(PT_PATH, "timing.json"), "w") as f:
        json.dump(all_time, f, indent=4)

    start_time = time.time()

    inverse_cell_type_mapping = {v: k for k, v in cell_type_mapping.items()}
    save_emb(test_emb, test_pred, test_label, PT_PATH, mode="test", cell_ids=test_obs.index.tolist())

    if len(test_emb) == 0:
        print(f"No test data for tissue {tissue_name}, skipping metric calculation.")
        return

    if len(test_pred) > 1:
        test_pred_flat = np.concatenate([np.ravel(x) for x in test_pred])
        test_label_flat = np.concatenate([np.ravel(x) for x in test_label])
    else:
        test_pred_flat = np.ravel(test_pred[0])
        test_label_flat = np.ravel(test_label[0])

    test_pred_flat = [inverse_cell_type_mapping[x] for x in test_pred_flat]
    test_label_flat = [inverse_cell_type_mapping[x] for x in test_label_flat]

    accuracy = accuracy_score(test_label_flat, test_pred_flat)
    f1 = f1_score(test_label_flat, test_pred_flat, average='macro')
    precision = precision_score(test_label_flat, test_pred_flat, average='macro')
    recall = recall_score(test_label_flat, test_label_flat if False else test_pred_flat, average='macro')  # keep your original
    metrics_time = time.time() - start_time
    print(f"Metrics calculation time: {metrics_time:.2f} seconds")
    all_time['metrics calculation'] = metrics_time

    total_time = time.time() - start
    with open(os.path.join(PT_PATH, "result.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Macro: {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"Total Time: {total_time:.2f} seconds\n")

    with open(os.path.join(PT_PATH, "timing.json"), "w") as f:
        json.dump(all_time, f, indent=4)


data_root = "/data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells"
# save_root = "/data/share/dungp/single-cell/ct-annot-results/CellFM_full_data_30ep"
save_root = './results'
os.makedirs(save_root, exist_ok=True)

log_file = f"{save_root}/run_CellFM_test_dataset.log"


def _run_one(job):
    """Return (status, tissue, message)."""
    save_root, tissue = job
    try:
        path_to_save = f"{save_root}/{tissue}"
        result_file = os.path.join(path_to_save, "result.txt")
        if os.path.exists(result_file):
            return ("skipped", tissue, "already done")

        run_dataset(save_root=save_root, tissue_name=tissue)
        return ("ok", tissue, "")
    except Exception as e:
        return ("error", tissue, f"{e}\n{traceback.format_exc()}")


def run_single_tissue(tissue: str):
    """Run exactly one tissue (good for external bash scheduling)."""
    tissue_path = os.path.join(data_root, tissue)
    if not os.path.isdir(tissue_path):
        raise FileNotFoundError(f"Tissue folder not found under data_root: {tissue_path}")

    train_path = os.path.join(tissue_path, "train.h5ad")
    test_path = os.path.join(tissue_path, "test.h5ad")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(f"train/test h5ad not found for tissue={tissue}: {train_path}, {test_path}")

    out_dir = os.path.join(save_root, tissue)
    result_file = os.path.join(out_dir, "result.txt")
    if os.path.exists(result_file):
        print(f"[single] Skipping {tissue} (already done): {result_file}")
        return

    print(f"[single] Running tissue={tissue} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    run_dataset(save_root=save_root, tissue_name=tissue)
    print(f"[single] Done tissue={tissue}")


def run_serial():
    tissues = []
    tissue_sizes = []
    for item in os.listdir(data_root):
        tissue_path = os.path.join(data_root, item)
        if os.path.isdir(tissue_path):
            train_path = os.path.join(tissue_path, 'train.h5ad')
            test_path = os.path.join(tissue_path, 'test.h5ad')
            if os.path.exists(train_path) and os.path.exists(test_path):
                tissue_size = os.path.getsize(train_path) + os.path.getsize(test_path)
                tissues.append(item)
                tissue_sizes.append(tissue_size)

    tissues = [t for _, t in sorted(zip(tissue_sizes, tissues), reverse=False)]

    error_count = 0
    with open(log_file, "a") as lf:
        for tissue in tqdm(tissues, total=len(tissues), desc="Tissues (serial)"):
            path_to_save = f"{save_root}/{tissue}"
            if os.path.exists(path_to_save + "/result.txt"):
                print(f"Skipping {tissue} as it is already done.")
                continue
            try:
                run_dataset(save_root=save_root, tissue_name=tissue)
                print(f"Done {tissue}")
            except Exception as e:
                error_count += 1
                msg = f"{e}\n{traceback.format_exc()}"
                lf.write(f"Error in tissue {tissue}: {msg}\n")
                lf.flush()
                print(f"Error in tissue {tissue}: {msg}")

    with open(log_file, "a") as f:
        f.write(f"Total error count: {error_count}\n")
    print(f"Total error count: {error_count}")


def run_parallel(max_workers: int = 4):
    tissues = []
    tissue_sizes = []
    for item in os.listdir(data_root):
        tissue_path = os.path.join(data_root, item)
        if os.path.isdir(tissue_path):
            train_path = os.path.join(tissue_path, 'train.h5ad')
            test_path = os.path.join(tissue_path, 'test.h5ad')
            if os.path.exists(train_path) and os.path.exists(test_path):
                tissue_size = os.path.getsize(train_path) + os.path.getsize(test_path)
                tissues.append(item)
                tissue_sizes.append(tissue_size)

    tissues = [t for _, t in sorted(zip(tissue_sizes, tissues), reverse=False)]

    error_count = 0
    tasks = [(save_root, t) for t in tissues if not os.path.exists(f"{save_root}/{t}/result.txt")]

    with open(log_file, "a") as lf:
        with mp.Pool(processes=max_workers) as pool:
            for status, tissue, msg in tqdm(
                pool.imap_unordered(_run_one, tasks, chunksize=1),
                total=len(tasks),
                desc=f"Tissues (parallel x{max_workers})"
            ):
                if status == "skipped":
                    print(f"Skipping {tissue} as it is already done.")
                elif status == "ok":
                    print(f"Done {tissue}")
                else:
                    error_count += 1
                    lf.write(f"Error in tissue {tissue}: {msg}\n")
                    lf.flush()
                    print(f"Error in tissue {tissue}: {msg}")

    with open(log_file, "a") as f:
        f.write(f"Total error count: {error_count}\n")
    print(f"Total error count: {error_count}")


# %%
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # NEW: single-tissue mode takes priority
    if args.tissue is not None:
        run_single_tissue(args.tissue)
    else:
        if args.parallel:
            run_parallel(max_workers=args.max_workers)
        else:
            run_serial()



"""
python run_CellFM_single_tissue.py --tissue "<TISSUE>" --device 0

python run_CellFM_single_tissue.py --tissue "dermis" --device 4


"""