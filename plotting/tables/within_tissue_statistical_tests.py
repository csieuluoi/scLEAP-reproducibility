#!/usr/bin/env python3
"""
Statistical analysis for within-tissue cell type annotation results.
Generates all numbers needed for the results section and supplementary tables.

Outputs (in outputs/tables/):
  - pairwise_wilcoxon_tests.csv        (scLEAD vs each method, paired Wilcoxon)
  - tissue_complexity_correlation.csv   (Spearman rho: n_cell_types vs F1)
  - complexity_binned_performance.csv   (mean±sd F1 per method per complexity bin)
  - frequency_binned_performance.csv    (mean±sd F1/Precision/Recall per frequency bin)
  - per_tissue_full_metrics.csv         (294 tissues × 6 methods, with metadata)
  - summary_statistics.csv             (mean/median/IQR/CV per method)
  - win_loss_matrix.csv                (tissue-level pairwise wins)
  - failure_analysis.csv               (tissues where scLEAD F1 < 0.30)
"""

import os
import pathlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = pathlib.Path(".")
DATA_ROOT = PROJECT_ROOT / "resources" / "scLEAP" / "results" / "tables"
OUT_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["scLEAD", "scTab", "CellFM", "scFoundation", "Random Forest", "lightGBM"]
METHOD_MAP = {"scLMCT": "scLEAD"}

# ============================================================
# Load data
# ============================================================
print("=" * 60)
print("Loading data...")
print("=" * 60)

cls_df = pd.read_csv(DATA_ROOT / "cell_type_classification.csv")
cls_df["method"] = cls_df["method"].replace(METHOD_MAP)
cls_df = cls_df[cls_df["method"].isin(METHODS)].copy()

# Exclude skeletal muscle tissue (same as plotting scripts)
cls_df = cls_df[cls_df["tissue"] != "skeletal muscle tissue"].copy()

counts_df = pd.read_csv(DATA_ROOT / "celltype_counts.csv")

bins_df = pd.read_csv(DATA_ROOT / "donor_celltype_bins_ALL_METHODS.csv")
bins_df["method"] = bins_df["method"].replace(METHOD_MAP)
bins_df = bins_df[bins_df["method"].isin(METHODS)].copy()

print(f"Classification data: {len(cls_df)} rows, {cls_df['tissue'].nunique()} tissues")
print(f"Cell counts data: {len(counts_df)} rows")
print(f"Frequency bins data: {len(bins_df)} rows")

# ============================================================
# 1. Summary statistics per method
# ============================================================
print("\n" + "=" * 60)
print("1. SUMMARY STATISTICS PER METHOD")
print("=" * 60)

summary_rows = []
for method in METHODS:
    mdf = cls_df[cls_df["method"] == method]
    for metric in ["f1", "precision", "recall", "mcc"]:
        vals = mdf[metric].dropna()
        summary_rows.append({
            "method": method,
            "metric": metric,
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "q25": vals.quantile(0.25),
            "q75": vals.quantile(0.75),
            "iqr": vals.quantile(0.75) - vals.quantile(0.25),
            "cv": vals.std() / vals.mean() if vals.mean() > 0 else np.nan,
            "min": vals.min(),
            "max": vals.max(),
            "n_tissues": len(vals),
            "n_above_0.8": (vals > 0.8).sum(),
            "n_above_0.5": (vals > 0.5).sum(),
            "n_below_0.5": (vals < 0.5).sum(),
            "n_below_0.3": (vals < 0.3).sum(),
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"), index=False)

# Print key numbers
for method in METHODS:
    f1_row = summary_df[(summary_df["method"] == method) & (summary_df["metric"] == "f1")].iloc[0]
    print(f"  {method:20s}  F1: mean={f1_row['mean']:.3f}, median={f1_row['median']:.3f}, "
          f"IQR=[{f1_row['q25']:.3f}, {f1_row['q75']:.3f}], CV={f1_row['cv']:.3f}")

# ============================================================
# 2. Paired Wilcoxon signed-rank tests (scLEAD vs each method)
# ============================================================
print("\n" + "=" * 60)
print("2. PAIRED WILCOXON TESTS (scLEAD vs each method)")
print("=" * 60)

sclead_df = cls_df[cls_df["method"] == "scLEAD"].set_index("tissue")
n_comparisons = 5  # Bonferroni correction

wilcoxon_rows = []
for method in METHODS:
    if method == "scLEAD":
        continue
    other_df = cls_df[cls_df["method"] == method].set_index("tissue")

    # Align on common tissues
    common = sclead_df.index.intersection(other_df.index)

    for metric in ["f1", "precision", "recall"]:
        x = sclead_df.loc[common, metric].values
        y = other_df.loc[common, metric].values

        # Remove pairs where both are NaN or identical
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        n_paired = len(x_clean)
        wins = (x_clean > y_clean).sum()
        ties = (x_clean == y_clean).sum()
        losses = (x_clean < y_clean).sum()

        # Wilcoxon signed-rank test
        diffs = x_clean - y_clean
        diffs_nonzero = diffs[diffs != 0]

        if len(diffs_nonzero) > 0:
            stat, p_value = stats.wilcoxon(diffs_nonzero, alternative="two-sided")
            # Rank-biserial correlation as effect size
            n_nz = len(diffs_nonzero)
            r_effect = 1 - (2 * stat) / (n_nz * (n_nz + 1) / 2)
        else:
            stat, p_value, r_effect = np.nan, 1.0, 0.0

        p_adjusted = min(p_value * n_comparisons, 1.0)

        wilcoxon_rows.append({
            "comparison": f"scLEAD vs {method}",
            "metric": metric,
            "n_tissues": n_paired,
            "W_statistic": stat,
            "p_value": p_value,
            "p_adjusted_bonferroni": p_adjusted,
            "effect_size_r": r_effect,
            "scLEAD_wins": wins,
            "ties": ties,
            "scLEAD_losses": losses,
            "mean_diff": np.mean(diffs_nonzero) if len(diffs_nonzero) > 0 else 0,
        })

wilcoxon_df = pd.DataFrame(wilcoxon_rows)
wilcoxon_df.to_csv(os.path.join(OUT_DIR, "pairwise_wilcoxon_tests.csv"), index=False)

print(f"\n{'Comparison':<25s} {'Metric':<10s} {'W':>10s} {'p-adj':>12s} {'r':>8s} {'Wins':>6s} {'Losses':>6s}")
print("-" * 80)
for _, row in wilcoxon_df.iterrows():
    print(f"{row['comparison']:<25s} {row['metric']:<10s} {row['W_statistic']:>10.0f} "
          f"{row['p_adjusted_bonferroni']:>12.2e} {row['effect_size_r']:>8.3f} "
          f"{row['scLEAD_wins']:>6.0f} {row['scLEAD_losses']:>6.0f}")

# ============================================================
# 3. Win/loss matrix (which method is best per tissue)
# ============================================================
print("\n" + "=" * 60)
print("3. WIN/LOSS ANALYSIS (best method per tissue)")
print("=" * 60)

pivot_f1 = cls_df.pivot_table(index="tissue", columns="method", values="f1")

# Strict first: method's F1 strictly greater than every other method's F1 on that tissue.
# Strict last: method's F1 strictly less than every other method's F1 on that tissue.
# Ties are excluded from both counts (e.g. the common_bile_duct case where all methods
# achieve F1 = 1.0 counts as neither a strict first nor a strict last for any method).
def _strict_rank_counts(pivot: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    first_counts = {m: 0 for m in pivot.columns}
    last_counts = {m: 0 for m in pivot.columns}
    for _, row in pivot.iterrows():
        vals = row.dropna()
        if len(vals) < 2:
            continue
        row_max = vals.max()
        row_min = vals.min()
        # Strict first: unique top scorer.
        max_methods = vals[vals == row_max].index
        if len(max_methods) == 1:
            first_counts[max_methods[0]] += 1
        # Strict last: unique bottom scorer.
        min_methods = vals[vals == row_min].index
        if len(min_methods) == 1:
            last_counts[min_methods[0]] += 1
    return pd.Series(first_counts), pd.Series(last_counts)


first_counts, last_counts = _strict_rank_counts(pivot_f1)
total_tissues = len(pivot_f1)
print("\nStrict first-place and last-place counts (F1):")
print(f"  Total tissues evaluated: {total_tissues}")
for method in METHODS:
    f_ct = int(first_counts.get(method, 0))
    l_ct = int(last_counts.get(method, 0))
    print(
        f"  {method:20s}: first={f_ct:4d} ({f_ct/total_tissues*100:.1f}%), "
        f"last={l_ct:4d} ({l_ct/total_tissues*100:.1f}%)"
    )

# Where scLEAD is NOT strictly first
not_best_mask = (pivot_f1["scLEAD"].notna()) & (pivot_f1.apply(lambda r: r["scLEAD"] != r.dropna().max() or (r.dropna() == r.dropna().max()).sum() > 1, axis=1))
not_best_count = int(not_best_mask.sum())
print(f"\nTissues where scLEAD is NOT strictly first: {not_best_count} ({not_best_count/total_tissues*100:.1f}%)")

win_counts_df = pd.DataFrame({
    "method": METHODS,
    "n_tissues_rank1": [int(first_counts.get(m, 0)) for m in METHODS],
    "pct_rank1": [first_counts.get(m, 0) / total_tissues * 100 for m in METHODS],
    "n_tissues_ranklast": [int(last_counts.get(m, 0)) for m in METHODS],
    "pct_ranklast": [last_counts.get(m, 0) / total_tissues * 100 for m in METHODS],
})
win_counts_df.to_csv(os.path.join(OUT_DIR, "win_loss_matrix.csv"), index=False)

# ============================================================
# 4. Tissue complexity analysis
# ============================================================
print("\n" + "=" * 60)
print("4. TISSUE COMPLEXITY ANALYSIS")
print("=" * 60)

# Count cell types per tissue (test set)
test_counts = counts_df[counts_df["set"] == "test"].copy()
tissue_complexity = test_counts.groupby("tissue")["cell_type"].nunique().reset_index()
tissue_complexity.columns = ["tissue", "n_cell_types"]

# Also count total test cells per tissue
tissue_test_cells = test_counts.groupby("tissue")["number_of_cells"].sum().reset_index()
tissue_test_cells.columns = ["tissue", "n_test_cells"]

tissue_meta = tissue_complexity.merge(tissue_test_cells, on="tissue")

# Merge with classification metrics
cls_with_complexity = cls_df.merge(tissue_meta, on="tissue", how="inner")

# Spearman correlation for each method
print("\nSpearman correlation (n_cell_types vs F1):")
corr_rows = []
for method in METHODS:
    mdf = cls_with_complexity[cls_with_complexity["method"] == method].dropna(subset=["f1"])
    if len(mdf) > 5:
        rho, p = stats.spearmanr(mdf["n_cell_types"], mdf["f1"])
        corr_rows.append({"method": method, "spearman_rho": rho, "p_value": p, "n": len(mdf)})
        print(f"  {method:20s}: rho={rho:.3f}, p={p:.2e}, n={len(mdf)}")

corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(os.path.join(OUT_DIR, "tissue_complexity_correlation.csv"), index=False)

# Complexity bins
bins = [0, 10, 25, 50, float("inf")]
labels = ["2-10", "11-25", "26-50", "51+"]
cls_with_complexity["complexity_bin"] = pd.cut(
    cls_with_complexity["n_cell_types"], bins=bins, labels=labels, right=True
)

complexity_rows = []
for method in METHODS:
    for bin_label in labels:
        subset = cls_with_complexity[
            (cls_with_complexity["method"] == method) &
            (cls_with_complexity["complexity_bin"] == bin_label)
        ]
        for metric in ["f1", "precision", "recall"]:
            vals = subset[metric].dropna()
            complexity_rows.append({
                "method": method,
                "complexity_bin": bin_label,
                "metric": metric,
                "mean": vals.mean() if len(vals) > 0 else np.nan,
                "std": vals.std() if len(vals) > 0 else np.nan,
                "median": vals.median() if len(vals) > 0 else np.nan,
                "n_tissues": len(vals),
            })

complexity_df = pd.DataFrame(complexity_rows)
complexity_df.to_csv(os.path.join(OUT_DIR, "complexity_binned_performance.csv"), index=False)

# Print complexity-stratified scLEAD vs scTab gap
print("\nscLEAD vs scTab F1 gap by tissue complexity:")
for bin_label in labels:
    sclead_mean = complexity_df[
        (complexity_df["method"] == "scLEAD") &
        (complexity_df["complexity_bin"] == bin_label) &
        (complexity_df["metric"] == "f1")
    ]["mean"].values
    sctab_mean = complexity_df[
        (complexity_df["method"] == "scTab") &
        (complexity_df["complexity_bin"] == bin_label) &
        (complexity_df["metric"] == "f1")
    ]["mean"].values
    if len(sclead_mean) > 0 and len(sctab_mean) > 0:
        gap = sclead_mean[0] - sctab_mean[0]
        n = complexity_df[
            (complexity_df["method"] == "scLEAD") &
            (complexity_df["complexity_bin"] == bin_label) &
            (complexity_df["metric"] == "f1")
        ]["n_tissues"].values[0]
        print(f"  {bin_label:>6s} cell types: scLEAD={sclead_mean[0]:.3f}, "
              f"scTab={sctab_mean[0]:.3f}, gap={gap:.3f} (n={n} tissues)")

# ============================================================
# 5. Frequency-binned performance
# ============================================================
print("\n" + "=" * 60)
print("5. CELL TYPE FREQUENCY ANALYSIS")
print("=" * 60)

freq_bins = ["<1%", "1-5%", "5-10%", ">10%"]

# Aggregate: mean across donors within tissue, then mean across tissues
for col in ["mean_f1", "mean_precision", "mean_recall"]:
    bins_df[col] = pd.to_numeric(bins_df[col], errors="coerce")

tissue_agg = bins_df.groupby(["method", "tissue", "cell_proportion"]).agg(
    mean_f1=("mean_f1", "mean"),
    mean_precision=("mean_precision", "mean"),
    mean_recall=("mean_recall", "mean"),
).reset_index()

freq_rows = []
for method in METHODS:
    for freq_bin in freq_bins:
        subset = tissue_agg[
            (tissue_agg["method"] == method) &
            (tissue_agg["cell_proportion"] == freq_bin)
        ]
        for metric_col, metric_name in [("mean_f1", "f1"), ("mean_precision", "precision"), ("mean_recall", "recall")]:
            vals = subset[metric_col].dropna()
            freq_rows.append({
                "method": method,
                "frequency_bin": freq_bin,
                "metric": metric_name,
                "mean": vals.mean() if len(vals) > 0 else np.nan,
                "std": vals.std() if len(vals) > 0 else np.nan,
                "n_tissues": len(vals),
            })

freq_df = pd.DataFrame(freq_rows)
freq_df.to_csv(os.path.join(OUT_DIR, "frequency_binned_performance.csv"), index=False)

print("\nMean F1 by cell type frequency:")
print(f"{'Method':<20s} {'<1%':>8s} {'1-5%':>8s} {'5-10%':>8s} {'>10%':>8s}")
print("-" * 55)
for method in METHODS:
    vals = []
    for fb in freq_bins:
        row = freq_df[(freq_df["method"] == method) & (freq_df["frequency_bin"] == fb) & (freq_df["metric"] == "f1")]
        vals.append(f"{row['mean'].values[0]:.3f}" if len(row) > 0 and not np.isnan(row["mean"].values[0]) else "  N/A")
    print(f"{method:<20s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s}")

# ============================================================
# 6. Per-tissue full metrics table (for supplementary)
# ============================================================
print("\n" + "=" * 60)
print("6. PER-TISSUE FULL METRICS TABLE")
print("=" * 60)

# Count donors per tissue (from counts_df)
train_counts = counts_df[counts_df["set"] == "train"]
donors_per_tissue_info = pd.DataFrame()  # We don't have donor info in counts_df directly

# Merge metrics with tissue metadata
per_tissue = cls_df.merge(tissue_meta, on="tissue", how="left")
per_tissue = per_tissue.sort_values(["tissue", "method"])
per_tissue.to_csv(os.path.join(OUT_DIR, "per_tissue_full_metrics.csv"), index=False)
print(f"Saved per-tissue table: {len(per_tissue)} rows")

# ============================================================
# 7. Failure analysis
# ============================================================
print("\n" + "=" * 60)
print("7. FAILURE ANALYSIS (scLEAD F1 < 0.30)")
print("=" * 60)

sclead_metrics = cls_df[cls_df["method"] == "scLEAD"].copy()
sclead_metrics = sclead_metrics.merge(tissue_meta, on="tissue", how="left")
failures = sclead_metrics[sclead_metrics["f1"] < 0.30].sort_values("f1")

print(f"\nTissues with scLEAD F1 < 0.30: {len(failures)}")
print(f"{'Tissue':<50s} {'F1':>6s} {'Prec':>6s} {'Rec':>6s} {'#Types':>6s}")
print("-" * 80)
for _, row in failures.iterrows():
    n_ct = int(row["n_cell_types"]) if not pd.isna(row.get("n_cell_types", np.nan)) else 0
    print(f"{row['tissue'][:50]:<50s} {row['f1']:>6.3f} {row['precision']:>6.3f} "
          f"{row['recall']:>6.3f} {n_ct:>6d}")

failures.to_csv(os.path.join(OUT_DIR, "failure_analysis.csv"), index=False)

# ============================================================
# 8. Additional: scLEAD ranking analysis
# ============================================================
print("\n" + "=" * 60)
print("8. scLEAD RANKING ANALYSIS")
print("=" * 60)

# Per tissue, rank methods by F1
rank_df = pivot_f1.rank(axis=1, ascending=False, method="min")
sclead_ranks = rank_df["scLEAD"].dropna()
print(f"\nscLEAD rank distribution across {len(sclead_ranks)} tissues:")
for rank in range(1, 7):
    count = (sclead_ranks == rank).sum()
    print(f"  Rank {rank}: {count} tissues ({count/len(sclead_ranks)*100:.1f}%)")

print(f"\nMean rank: {sclead_ranks.mean():.2f}")
print(f"Median rank: {sclead_ranks.median():.1f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE. All outputs saved to:", OUT_DIR)
print("=" * 60)
