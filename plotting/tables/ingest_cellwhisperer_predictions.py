#!/usr/bin/env python3
"""Ingest CellWhisperer per-cell zero-shot predictions into the canonical
per-method CSV schema.

Reads (read-only):
    dung-code/zero_shot_results/cellwhisperer_results/<dataset>[.project]/
        cellwhisperer_clip_v1/similarity_scores_per_cell.csv
    dung-code/resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv

Writes:
    outputs/tables/method_predictions/CellWhisperer.csv
    outputs/tables/label_alias/cellwhisperer_<dataset>.yaml

CellWhisperer's `similarity_scores_per_cell.csv` already stores predictions in
Cell Ontology CL name space (e.g. ``pancreatic A cell``). We just lowercase
the strings to match scLEAD's CL truth and verify a clean bijection per
dataset the same way we do for LangCell. Any edge cases where CellWhisperer's
label differs from scLEAD's beyond case (e.g. ``Henle`` vs ``henle``) are
surfaced so the alias map can be reviewed.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import yaml


ROOT = pathlib.Path(".")
SCLEAD_CSV = ROOT / "resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv"
CW_ROOT = pathlib.Path("./resources/zero_shot_results/cellwhisperer_results")
OUT_CSV = ROOT / "outputs/tables/method_predictions/CellWhisperer.csv"
ALIAS_DIR = ROOT / "outputs/tables/label_alias"

DATASETS = [
    ("E-MTAB-9221", "E-MTAB-9221.project"),
    ("GSE185948",   "GSE185948"),
    ("E-HCAD-9",    "E-HCAD-9.project"),
    ("E-CURD-114",  "E-CURD-114.project"),
    ("E-HCAD-15",   "E-HCAD-15.project"),
    ("E-MTAB-5061", "E-MTAB-5061"),
    ("GSE81608",    "GSE81608"),
    ("GSE84133",    "GSE84133"),
    ("GSE86473",    "GSE86473"),
    ("GSE111976",   "GSE111976"),
]


def norm(s: str) -> str:
    """Lowercase + collapse whitespace. Match scLEAD's cell_type format."""
    return " ".join(str(s).split()).lower()


def derive_alias(cw_truth: pd.Series, sc_truth: pd.Series) -> dict[str, str]:
    """Build CellWhisperer-native-label -> scLEAD CL-lowercase map, row-aligned."""
    pairs = pd.DataFrame({"native": cw_truth, "cl": sc_truth})
    groups = pairs.groupby("native")["cl"].unique()
    conflicts = {k: v.tolist() for k, v in groups.items() if len(v) > 1}
    if conflicts:
        raise ValueError(f"non-bijective native->CL map: {conflicts}")
    return {k: v[0] for k, v in groups.items()}


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ALIAS_DIR.mkdir(parents=True, exist_ok=True)

    sclead_all = pd.read_csv(SCLEAD_CSV)
    sclead_all["dataset"] = sclead_all["dataset"].str.replace(r"\.project$", "",
                                                               regex=True)

    rows_out = []
    for scl_id, cw_id in DATASETS:
        cw_path = CW_ROOT / cw_id / "cellwhisperer_clip_v1" / "similarity_scores_per_cell.csv"
        if not cw_path.exists():
            raise FileNotFoundError(cw_path)
        cw = pd.read_csv(cw_path)
        sc = sclead_all[sclead_all["dataset"] == scl_id].reset_index(drop=True)

        if len(cw) != len(sc):
            raise AssertionError(
                f"{scl_id}: row-count mismatch CellWhisperer={len(cw)} scLEAD={len(sc)}"
            )

        # Derive alias: CellWhisperer truth (original casing) -> scLEAD CL truth.
        alias = derive_alias(cw["true_label"], sc["cell_type"])
        with open(ALIAS_DIR / f"cellwhisperer_{scl_id}.yaml", "w") as f:
            yaml.safe_dump(alias, f, sort_keys=True)

        pred_cl = cw["pred_label"].map(alias)
        unmapped = sorted(cw.loc[pred_cl.isna(), "pred_label"].unique().tolist())
        # For any unmapped predictions, fall back to lower-case normalisation.
        if unmapped:
            fallback = {u: norm(u) for u in unmapped}
            pred_cl = pred_cl.fillna(cw["pred_label"].map(fallback))

        agree = int((pred_cl == sc["cell_type"]).sum())
        print(
            f"{scl_id:14s}  n={len(sc):>7,}  native_classes={len(alias):>3d}  "
            f"unmapped_preds_fallback={len(unmapped):>2d}  "
            f"agree={agree:>7,} ({agree / len(sc):.3f})"
        )
        if unmapped:
            print(f"    unmapped (used lower-cased fallback): {unmapped}")

        rows_out.append(pd.DataFrame({
            "dataset": scl_id,
            "cell_id": sc["cell_id"].to_numpy(),
            "cell_type": sc["cell_type"].to_numpy(),
            "predicted_cell_type": pred_cl.to_numpy(),
        }))

    out = pd.concat(rows_out, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV} ({len(out):,} rows, {out['dataset'].nunique()} datasets)")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()
