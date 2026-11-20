#!/usr/bin/env python3
"""
Extract per-tissue runtime from training/inference logs under dung-code/
for both the Figure 1 classification methods and the Figure 3 clustering methods.

Outputs:
  outputs/tables/runtime_classification.csv
  outputs/tables/runtime_clustering.csv
"""

from __future__ import annotations

import json
import pathlib
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = pathlib.Path(".")
CLASS_ROOT = pathlib.Path("./resources/annotation_results")
CLUST_ROOT = pathlib.Path("./resources/clustering_results")
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_kv_text(path: Path) -> dict:
    """Parse `key: value` text files into a dict. Values are stripped strings."""
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def extract_seconds(value: str) -> float | None:
    """Pull the first number out of strings like '168.24 sec' or '1305.32'."""
    if value is None:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", value)
    return float(m.group()) if m else None


# ---------- classification (Fig 1 methods) ----------

def parse_scLEAD(tissue_dir: Path):
    f = tissue_dir / "time_log.txt"
    if not f.exists():
        return None
    kv = parse_kv_text(f)
    data = extract_seconds(kv.get("Data processing time"))
    train = extract_seconds(kv.get("Model creating and training time"))
    # Inference is split across several keys; sum what's present.
    inf_keys = ["Latent inference time", "Text embedding time",
                "FAISS build time", "Predict time"]
    inf_vals = [extract_seconds(kv.get(k)) for k in inf_keys]
    inference = sum(v for v in inf_vals if v is not None) if any(inf_vals) else None
    total = extract_seconds(kv.get("Total time"))
    return dict(data_proc_sec=data, train_sec=train,
                inference_sec=inference, total_sec=total, source=str(f))


def parse_scTab(tissue_dir: Path):
    f = tissue_dir / "time_log.txt"
    if not f.exists():
        return None
    kv = parse_kv_text(f)
    return dict(
        data_proc_sec=extract_seconds(kv.get("Data processing time")),
        train_sec=extract_seconds(kv.get("Model training time")),
        inference_sec=extract_seconds(kv.get("Inference time")),
        total_sec=extract_seconds(kv.get("Total time")),
        source=str(f),
    )


def parse_CellFM(tissue_dir: Path):
    f = tissue_dir / "timing.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    # Values are seconds (confirmed by magnitude on sample tissues)
    train = d.get("fine-tuning")
    data = d.get("data processing")
    inference = d.get("evaluation")
    total = sum(v for v in d.values() if isinstance(v, (int, float)))
    return dict(data_proc_sec=data, train_sec=train, inference_sec=inference,
                total_sec=total, source=str(f))


def parse_result_txt_total_only(tissue_dir: Path):
    f = tissue_dir / "result.txt"
    if not f.exists():
        return None
    kv = parse_kv_text(f)
    total = extract_seconds(kv.get("Total Time"))
    if total is None:
        return None
    return dict(data_proc_sec=None, train_sec=None, inference_sec=None,
                total_sec=total, source=str(f))


CLASS_METHODS = {
    "scLEAD":       ("scLEAD",       parse_scLEAD),
    "scTab":        ("scTab",        parse_scTab),
    "CellFM":       ("CellFM",       parse_CellFM),
    "scFoundation": ("scFoundation", parse_result_txt_total_only),
    "RF":           ("Random Forest", parse_result_txt_total_only),
    "LGBM":         ("LightGBM",     parse_result_txt_total_only),
}


def build_classification_csv():
    rows = []
    for dirname, (display, parser) in CLASS_METHODS.items():
        root = CLASS_ROOT / dirname
        if not root.is_dir():
            print(f"WARN: {root} missing")
            continue
        for tissue_dir in sorted(root.iterdir()):
            if not tissue_dir.is_dir():
                continue
            if tissue_dir.name.startswith("_"):
                continue
            parsed = parser(tissue_dir)
            if parsed is None:
                continue
            rows.append(dict(method=display, tissue=tissue_dir.name, **parsed))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "runtime_classification.csv", index=False)
    print(f"Wrote {OUT_DIR/'runtime_classification.csv'} ({len(df)} rows)")
    print(df.groupby("method").size().to_string())
    return df


# ---------- clustering (Fig 3 methods) ----------

CLUST_METHODS = {
    "scLEAD":   ("scLEAD",   "Time"),
    "scanpy":   ("SCANPY",   "Time"),
    "seurat":   ("Seurat",   "Time"),
    "sharp":    ("SHARP",    "Time"),
    "pegasus":  ("Pegasus",  "Time"),
    "monocle3": ("Monocle3", "Time"),
    "scvi":     ("scVI",     "Time_sec"),
}


def build_clustering_csv():
    rows = []
    for dirname, (display, time_key) in CLUST_METHODS.items():
        root = CLUST_ROOT / dirname
        if not root.is_dir():
            print(f"WARN: {root} missing")
            continue
        for tissue_dir in sorted(root.iterdir()):
            if not tissue_dir.is_dir():
                continue
            if tissue_dir.name.startswith("_"):
                continue
            f = tissue_dir / "results.txt"
            if not f.exists():
                continue
            kv = parse_kv_text(f)
            total = extract_seconds(kv.get(time_key))
            if total is None:
                continue
            rows.append(dict(method=display, tissue=tissue_dir.name,
                             total_sec=total,
                             source=str(f)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "runtime_clustering.csv", index=False)
    print(f"Wrote {OUT_DIR/'runtime_clustering.csv'} ({len(df)} rows)")
    print(df.groupby("method").size().to_string())
    return df


if __name__ == "__main__":
    print("=== Classification runtimes ===")
    build_classification_csv()
    print()
    print("=== Clustering runtimes ===")
    build_clustering_csv()
