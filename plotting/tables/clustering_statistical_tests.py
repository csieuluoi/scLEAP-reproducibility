#!/usr/bin/env python3
"""
Statistical analysis for clustering and visualization results (Figure 3).
Outputs (in outputs/tables/):
  - clustering_summary_statistics.csv
  - clustering_pairwise_wilcoxon.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = "."
DATA_ROOT = os.path.join(PROJECT_ROOT, "resources", "scLEAP", "results", "tables")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "tables")
os.makedirs(OUT_DIR, exist_ok=True)
METHODS = ["scLEAD", "scanpy", "Seurat", "SHARP", "scVI", "pegasus", "monocle3"]
METHOD_MAP = {"scLMCT": "scLEAD"}
METRICS = ["nmi", "ari", "completeness"]

df = pd.read_csv(os.path.join(DATA_ROOT, "cell_type_clustering.csv"))
df["method"] = df["method"].replace(METHOD_MAP)
df = df[df["method"].isin(METHODS)].copy()

print("=" * 60)
print("1. SUMMARY STATISTICS")
print("=" * 60)

rows = []
for method in METHODS:
    mdf = df[df["method"] == method]
    for metric in METRICS:
        vals = mdf[metric].dropna()
        rows.append({
            "method": method, "metric": metric,
            "mean": vals.mean(), "median": vals.median(), "std": vals.std(),
            "q25": vals.quantile(0.25), "q75": vals.quantile(0.75),
            "iqr": vals.quantile(0.75) - vals.quantile(0.25),
            "n": len(vals),
        })

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(OUT_DIR, "clustering_summary_statistics.csv"), index=False)

for metric in METRICS:
    print(f"\n  {metric.upper()}:")
    for method in METHODS:
        r = summary[(summary["method"] == method) & (summary["metric"] == metric)].iloc[0]
        print(f"    {method:12s}  median={r['median']:.3f}  mean={r['mean']:.3f}  IQR=[{r['q25']:.3f}, {r['q75']:.3f}]")

print("\n" + "=" * 60)
print("2. PAIRED WILCOXON TESTS (scLEAD vs each method)")
print("=" * 60)

sclead = df[df["method"] == "scLEAD"].set_index("tissue")
n_comparisons = 6

wilcoxon_rows = []
for method in METHODS:
    if method == "scLEAD":
        continue
    other = df[df["method"] == method].set_index("tissue")
    common = sclead.index.intersection(other.index)

    for metric in METRICS:
        x = sclead.loc[common, metric].values
        y = other.loc[common, metric].values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_c, y_c = x[mask], y[mask]
        diffs = x_c - y_c
        diffs_nz = diffs[diffs != 0]

        wins = (x_c > y_c).sum()
        losses = (x_c < y_c).sum()
        ties = (x_c == y_c).sum()

        if len(diffs_nz) > 0:
            stat, p = stats.wilcoxon(diffs_nz, alternative="two-sided")
            n_nz = len(diffs_nz)
            r_effect = 1 - (2 * stat) / (n_nz * (n_nz + 1) / 2)
        else:
            stat, p, r_effect = np.nan, 1.0, 0.0

        wilcoxon_rows.append({
            "comparison": f"scLEAD vs {method}", "metric": metric,
            "n": len(x_c), "W": stat,
            "p_value": p, "p_adj": min(p * n_comparisons, 1.0),
            "effect_r": r_effect, "wins": wins, "losses": losses, "ties": ties,
        })

wilcoxon_df = pd.DataFrame(wilcoxon_rows)
wilcoxon_df.to_csv(os.path.join(OUT_DIR, "clustering_pairwise_wilcoxon.csv"), index=False)

print(f"\n{'Comparison':<25s} {'Metric':<15s} {'p-adj':>12s} {'r':>8s} {'Wins':>6s} {'Losses':>6s}")
print("-" * 75)
for _, r in wilcoxon_df.iterrows():
    print(f"{r['comparison']:<25s} {r['metric']:<15s} {r['p_adj']:>12.2e} {r['effect_r']:>8.3f} {r['wins']:>6.0f} {r['losses']:>6.0f}")

print("\nDone.")
