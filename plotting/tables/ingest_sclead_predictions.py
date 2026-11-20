#!/usr/bin/env python3
"""Ingest scLEAD zero-shot predictions into the canonical per-method CSV.

Reads (read-only):
    dung-code/resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv

Writes:
    outputs/tables/method_predictions/scLEAD.csv

The dung-code master CSV stores scLEAD's per-cell zero-shot predictions for every
external evaluation dataset, with columns `dataset, cell_id, cell_type,
predicted_cell_type`. Some dataset names carry an `.project` suffix from the
upstream ArrayExpress preprocessing. We strip that suffix so the canonical CSV
matches the dataset keys used by `compute_zeroshot_bootstrap.py` and the
LangCell / CellWhisperer ingest scripts.

Scope: every dataset present in the master CSV (currently 10).
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd


ROOT = pathlib.Path(".")
SCLEAD_CSV = ROOT / "resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv"
OUT_CSV = ROOT / "outputs/tables/method_predictions/scLEAD.csv"


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SCLEAD_CSV)
    df["dataset"] = df["dataset"].str.replace(r"\.project$", "", regex=True)

    df = df[["dataset", "cell_id", "cell_type", "predicted_cell_type"]]
    df.to_csv(OUT_CSV, index=False)

    counts = df.groupby("dataset").size().sort_values(ascending=False)
    print(f"Wrote {OUT_CSV} ({len(df):,} rows, {df['dataset'].nunique()} datasets)")
    for ds, n in counts.items():
        print(f"  {ds:14s}  n={n:>7,}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
