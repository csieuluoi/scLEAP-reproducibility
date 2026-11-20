#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from data_preprocess.utils import BertModel as MedBertModel
from data_preprocess.utils import LangCellDataCollatorForCellClassification as DataCollatorForCellClassification


class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj, map_location="cpu"))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


def load_candidate_labels(path: str):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def dedup_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_texts(types, type2text_path=None):
    if type2text_path is None:
        return [f"cell type: {t}" for t in types]

    with open(type2text_path, "r") as f:
        type2text = json.load(f)

    texts = []
    missing = []
    for t in types:
        if t in type2text:
            texts.append(type2text[t])
        else:
            missing.append(t)
            texts.append(f"cell type: {t}")

    if missing:
        print(f"[WARN] Missing {len(missing)} labels in type2text_json; used fallback for examples like: {missing[:10]}")
    return texts


def text_encode(texts, tokenizer, text_encoder, device):
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    text = text_encoder(**tok).pooler_output
    return text


def cell_encode(cell_model, cell_input_ids, cell_atts, device):
    cell = cell_model(cell_input_ids.to(device), cell_atts.to(device))
    cell_last_h = cell.last_hidden_state
    cell_pooler = cell.pooler_output
    return cell_last_h, cell_pooler


def ctm(texts, tokenizer, text_encoder, ctm_head, cell_emb, cell_atts, device):
    tok = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    output = text_encoder(
        **tok,
        encoder_hidden_states=cell_emb.to(device),
        encoder_attention_mask=cell_atts.to(device),
        return_dict=True,
        mode="multimodal",
    )
    logits = ctm_head(output.last_hidden_state[:, 0, :])
    logits = F.softmax(logits, dim=-1)[..., 1]
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized_path", required=True)
    parser.add_argument(
        "--candidate_labels",
        default=None,
        help="Optional. If omitted, use unique celltype values from the dataset."
    )
    parser.add_argument(
        "--type2text_json",
        default=None,
        help="Optional. If missing, fallback to 'cell type: {}'."
    )
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_pt", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")

    ckpt_dir = Path(args.ckpt_dir)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    if args.output_pt:
        Path(args.output_pt).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    dataset = load_from_disk(args.tokenized_path)
    print(f"[INFO] loaded dataset: n={len(dataset)}")

    dataset_sub = dataset

    label_name = None
    for cand in ["celltype", "cell_type", "str_labels", "labels"]:
        if cand in dataset_sub.column_names:
            label_name = cand
            break

    if label_name is None:
        raise ValueError(
            f"No label column found among: ['celltype', 'cell_type', 'str_labels', 'labels'] "
            f"in {dataset_sub.column_names}"
        )

    if label_name != "celltype":
        dataset_sub = dataset_sub.rename_column(label_name, "celltype")

    true_labels_raw = np.array(dataset_sub["celltype"]).astype(str)

    # ------------------------------------------------------------
    # Candidate types
    # ------------------------------------------------------------
    if args.candidate_labels is None:
        types = dedup_preserve_order(dataset_sub["celltype"])
    else:
        types = load_candidate_labels(args.candidate_labels)
        types = dedup_preserve_order(types)

    texts = build_texts(types, args.type2text_json)

    print(f"[INFO] number of candidate types={len(types)}")
    print(f"[INFO] first few candidate types={types[:10]}")
    print(f"[INFO] first few texts={texts[:3]}")

    # ------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------
    model = BertModel.from_pretrained(str(ckpt_dir / "cell_bert"))
    model.pooler = Pooler(
        model.config,
        pretrained_proj=str(ckpt_dir / "cell_proj.bin"),
        proj_dim=256
    )
    model = model.to(device)

    text_pretrained_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = BertTokenizer.from_pretrained(text_pretrained_model)
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

    text_encoder = MedBertModel.from_pretrained(
        str(ckpt_dir / "text_bert"),
        add_pooling_layer=True
    )
    text_encoder.pooler = Pooler(
        text_encoder.config,
        pretrained_proj=str(ckpt_dir / "text_proj.bin"),
        proj_dim=256
    )
    text_encoder = text_encoder.to(device)

    ctm_head = nn.Linear(text_encoder.config.hidden_size, 2)
    ctm_head.load_state_dict(torch.load(ckpt_dir / "ctm_head.bin", map_location="cpu"))
    ctm_head = ctm_head.to(device)

    # ------------------------------------------------------------
    # Encode texts
    # ------------------------------------------------------------
    with torch.no_grad():
        text_embs = torch.cat(
            [text_encode([text], tokenizer, text_encoder, device) for text in texts],
            dim=0
        ).T.contiguous().to(device)

    text_embs.requires_grad = False
    type2num = {t: i for i, t in enumerate(types)}

    # ------------------------------------------------------------
    # Prepare dataset
    # ------------------------------------------------------------
    def classes_to_ids(example):
        example["label"] = type2num[str(example["celltype"])]
        return example

    testdataset = dataset_sub.map(classes_to_ids, num_proc=1)

    remove_columns = list(testdataset.column_names)
    remove_columns.remove("input_ids")
    remove_columns.remove("label")
    testdataset = testdataset.remove_columns(remove_columns)

    labels = torch.tensor(testdataset["label"])
    collator = DataCollatorForCellClassification()
    dataloader = DataLoader(
        testdataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False
    )

    # ------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------
    n = len(dataset_sub)
    num_types = text_embs.shape[-1]

    cell_embs = torch.zeros(n, 256)
    sim_logits = torch.zeros(n, num_types)
    ctm_logits = torch.zeros(n, num_types)
    logits = torch.zeros(n, num_types)
    preds = torch.zeros(n, dtype=torch.long)

    model.eval()
    text_encoder.eval()
    ctm_head.eval()

    with torch.no_grad():
        for i, d in tqdm(enumerate(dataloader), total=(n + args.batch_size - 1) // args.batch_size):
            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, n)
            bs = end - start

            cell_last_h, cellemb = cell_encode(model, d["input_ids"], d["attention_mask"], device)

            sim = (cellemb @ text_embs) / 0.05
            sim_logit = F.softmax(sim, dim=-1)

            ctm_logit = torch.zeros_like(sim_logit)
            for text_idx, text in enumerate(texts):
                text_list = [text] * sim_logit.shape[0]
                ctm_logit[:, text_idx] = ctm(
                    text_list,
                    tokenizer,
                    text_encoder,
                    ctm_head,
                    cell_last_h,
                    d["attention_mask"],
                    device
                )

            ctm_logit = F.softmax(ctm_logit, dim=-1)

            logit = (sim_logit + ctm_logit) / 2
            pred = logit.argmax(dim=-1)

            sim_logits[start:end] = sim_logit[:bs].cpu()
            ctm_logits[start:end] = ctm_logit[:bs].cpu()
            logits[start:end] = logit[:bs].cpu()
            cell_embs[start:end] = cellemb[:bs].cpu()
            preds[start:end] = pred[:bs].cpu()

    pred_labels = [types[i] for i in preds.tolist()]

    # ------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------
    out = pd.DataFrame({
        "pred_index": preds.numpy(),
        "pred_label": pred_labels,
        "max_logit": logits.max(dim=-1).values.numpy(),
        "max_sim_prob": sim_logits.max(dim=-1).values.numpy(),
        "max_ctm_prob": ctm_logits.max(dim=-1).values.numpy(),
        "true_index": labels.numpy(),
        "true_label": true_labels_raw,
    })

    out.to_csv(args.output_csv, index=False)
    print(f"[INFO] saved predictions to {args.output_csv}")

    if args.output_pt:
        torch.save({
            "types": types,
            "texts": texts,
            "preds": preds,
            "labels": labels,
            "logits": logits,
            "sim_logits": sim_logits,
            "ctm_logits": ctm_logits,
            "cell_embs": cell_embs,
            "true_label": true_labels_raw,
        }, args.output_pt)
        print(f"[INFO] saved tensors to {args.output_pt}")


if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=7 python run_langcell_zeroshot_no_mappings.py \
  --tokenized_path /data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/LangCell/data/tokenized_dataset/GSE111976 \
  --candidate_labels /data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/LangCell/data/tokenized_dataset/GSE111976/candidate_labels.txt \
  --ckpt_dir ./ckpt \
  --output_csv ./results/GSE111976_preds.csv \
  --output_pt ./results/GSE111976_outputs.pt
"""