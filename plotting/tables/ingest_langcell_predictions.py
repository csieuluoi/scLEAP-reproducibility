#!/usr/bin/env python3
"""Ingest LangCell zero-shot predictions into the canonical per-method CSV.

Reads (read-only):
    dung-code/zero_shot_results/langcell_results/<dataset>[.project]_preds.csv
    dung-code/resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv

Writes:
    outputs/tables/method_predictions/LangCell.csv
    outputs/tables/label_alias/langcell_<dataset>.yaml

LangCell keeps each dataset's native annotations (e.g. "Alpha", "MHC class II")
in its `true_label` and `pred_label` columns. scLEAD's preprocessing normalises
those same cells onto Cell Ontology lowercase strings ("pancreatic a cell",
"mhc class ii-positive cell"). Row order is 1-to-1 identical between the two
outputs on every dataset, so we derive the native-to-CL alias map from
`dict(zip(langcell.true_label, sclead.cell_type))` and translate LangCell's
`pred_label` through that same map.

Scope: the 9 scLEAD zero-shot datasets only.
"""
from __future__ import annotations

import pathlib
import sys

import pandas as pd
import yaml


ROOT = pathlib.Path(".")
SCLEAD_CSV = ROOT / "resources/scLEAP/results/tables/Fig2_zero_shot_all_cells_truth_pred_dataset.csv"
LANGCELL_DIR = pathlib.Path("./resources/zero_shot_results/langcell_results")
OUT_CSV = ROOT / "outputs/tables/method_predictions/LangCell.csv"
ALIAS_DIR = ROOT / "outputs/tables/label_alias"

DATASETS = [
    "E-MTAB-9221", "GSE185948", "E-HCAD-9", "E-CURD-114", "E-HCAD-15",
    "E-MTAB-5061", "GSE81608", "GSE84133", "GSE86473", "GSE111976",
]


def langcell_csv_path(dataset: str) -> pathlib.Path:
    for stem in (f"{dataset}.project_preds.csv", f"{dataset}_preds.csv"):
        p = LANGCELL_DIR / stem
        if p.exists():
            return p
    raise FileNotFoundError(f"no LangCell preds for {dataset}")


def derive_alias(langcell_truth: pd.Series, sclead_truth: pd.Series) -> dict[str, str]:
    """Build native->CL map, requiring a clean bijection across rows."""
    pairs = pd.DataFrame({"native": langcell_truth, "cl": sclead_truth})
    groups = pairs.groupby("native")["cl"].unique()
    conflicts = {k: v.tolist() for k, v in groups.items() if len(v) > 1}
    if conflicts:
        raise ValueError(f"non-bijective native->CL mapping: {conflicts}")
    return {k: v[0] for k, v in groups.items()}


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ALIAS_DIR.mkdir(parents=True, exist_ok=True)

    sclead_all = pd.read_csv(SCLEAD_CSV)
    sclead_all["dataset"] = sclead_all["dataset"].str.replace(r"\.project$", "", regex=True)

    rows_out: list[pd.DataFrame] = []
    for ds in DATASETS:
        lc = pd.read_csv(langcell_csv_path(ds))
        sc = sclead_all[sclead_all["dataset"] == ds].reset_index(drop=True)

        if len(lc) != len(sc):
            raise AssertionError(
                f"{ds}: row count mismatch LangCell={len(lc)} scLEAD={len(sc)}"
            )

        alias = derive_alias(lc["true_label"], sc["cell_type"])
        with open(ALIAS_DIR / f"langcell_{ds}.yaml", "w") as f:
            yaml.safe_dump(alias, f, sort_keys=True)

        unmapped = sorted(set(lc["pred_label"]) - set(alias))
        pred_cl = lc["pred_label"].map(alias).fillna(lc["pred_label"])

        agree = int((pred_cl == sc["cell_type"]).sum())
        print(
            f"{ds:14s}  n={len(sc):>7,}  native_classes={len(alias):>3d}  "
            f"unmapped_preds={len(unmapped):>2d}  agree={agree:>7,} "
            f"({agree / len(sc):.3f})"
        )
        if unmapped:
            print(f"    unmapped in {ds}: {unmapped}")

        rows_out.append(pd.DataFrame({
            "dataset": ds,
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
