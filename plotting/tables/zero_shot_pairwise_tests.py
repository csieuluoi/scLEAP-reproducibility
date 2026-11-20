#!/usr/bin/env python3
"""
Pairwise Wilcoxon signed-rank tests for zero-shot annotation (scLEAD vs each
competitor) over the 49 evaluated cell types.

Inputs (read-only):
  outputs/tables/zeroshot_per_celltype_bootstrap.csv
     columns include method, cell_type, F1_observed

Outputs:
  outputs/tables/zeroshot_pairwise_wilcoxon.csv
    One row per (competitor, metric). Columns:
      comparison, metric, n_cell_types, W_statistic, p_value,
      p_adjusted_bonferroni, effect_size_r, scLEAD_wins, ties, scLEAD_losses,
      mean_diff
"""

import os
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = pathlib.Path(".")
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
IN_CSV = OUT_DIR / "zeroshot_per_celltype_bootstrap.csv"

COMPETITORS = ["LangCell", "CellWhisperer"]
METRICS = ["F1_observed", "Precision_observed", "Recall_observed"]
METRIC_LABEL = {"F1_observed": "F1", "Precision_observed": "Precision", "Recall_observed": "Recall"}

N_COMPARISONS = 2  # scLEAD vs LangCell, scLEAD vs CellWhisperer

df = pd.read_csv(IN_CSV)
scl = df[df["method"] == "scLEAD"].set_index("cell_type")

rows = []
for comp in COMPETITORS:
    other = df[df["method"] == comp].set_index("cell_type")
    common = scl.index.intersection(other.index)
    for metric in METRICS:
        x = scl.loc[common, metric].values
        y = other.loc[common, metric].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean, y_clean = x[mask], y[mask]
        n_paired = len(x_clean)

        wins = int((x_clean > y_clean).sum())
        ties = int((x_clean == y_clean).sum())
        losses = int((x_clean < y_clean).sum())

        diffs = x_clean - y_clean
        diffs_nz = diffs[diffs != 0]
        if len(diffs_nz) > 0:
            stat, p_value = stats.wilcoxon(diffs_nz, alternative="two-sided")
            n_nz = len(diffs_nz)
            r_effect = 1 - (2 * stat) / (n_nz * (n_nz + 1) / 2)
        else:
            stat, p_value, r_effect = np.nan, 1.0, 0.0
        p_adj = min(p_value * N_COMPARISONS, 1.0)

        rows.append({
            "comparison": f"scLEAD vs {comp}",
            "metric": METRIC_LABEL[metric],
            "n_cell_types": n_paired,
            "W_statistic": stat,
            "p_value": p_value,
            "p_adjusted_bonferroni": p_adj,
            "effect_size_r": r_effect,
            "scLEAD_wins": wins,
            "ties": ties,
            "scLEAD_losses": losses,
            "mean_diff": float(np.mean(diffs_nz)) if len(diffs_nz) > 0 else 0.0,
        })

out = pd.DataFrame(rows)
out_path = OUT_DIR / "zeroshot_pairwise_wilcoxon.csv"
out.to_csv(out_path, index=False)

print(f"{'Comparison':<25s} {'Metric':<11s} {'n':>4s} {'W':>8s} {'p-adj':>12s} {'r':>7s} {'W/T/L':>12s}")
print("-" * 90)
for _, r in out.iterrows():
    wtl = f"{r['scLEAD_wins']}/{r['ties']}/{r['scLEAD_losses']}"
    print(f"{r['comparison']:<25s} {r['metric']:<11s} {int(r['n_cell_types']):>4d} "
          f"{r['W_statistic']:>8.1f} {r['p_adjusted_bonferroni']:>12.2e} "
          f"{r['effect_size_r']:>7.3f} {wtl:>12s}")
print(f"\nWrote {out_path}")
