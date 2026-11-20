#!/usr/bin/env Rscript
# Build runtime summary tables for supplementary.
# Reads outputs/tables/runtime_classification.csv and runtime_clustering.csv
# and prints two LaTeX tabular blocks to stdout + writes them to
# outputs/tables/runtime_tables.tex for easy copy-paste.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = FALSE)

script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))

project_root <- "./"

out_dir <- file.path(project_root, "outputs", "tables")

# Display units. Seconds if < 60, minutes if < 3600, hours otherwise.
fmt_time <- function(sec) {
  ifelse(is.na(sec), "--",
    ifelse(sec < 60,   sprintf("%.1f s",   sec),
    ifelse(sec < 3600, sprintf("%.1f min", sec / 60),
                       sprintf("%.2f h",   sec / 3600))))
}

summarise_method <- function(df, time_col = "total_sec") {
  df %>%
    group_by(method) %>%
    summarise(
      n           = sum(!is.na(.data[[time_col]])),
      median      = median(.data[[time_col]], na.rm = TRUE),
      q25         = quantile(.data[[time_col]], 0.25, na.rm = TRUE),
      q75         = quantile(.data[[time_col]], 0.75, na.rm = TRUE),
      total_sec   = sum(.data[[time_col]], na.rm = TRUE),
      .groups = "drop"
    ) %>%
    mutate(
      median_fmt = fmt_time(median),
      iqr_fmt    = sprintf("%s -- %s", fmt_time(q25), fmt_time(q75)),
      total_h    = total_sec / 3600
    )
}

# ============================================================
# DISPLAY NAME HELPER
# ============================================================

display_method <- function(x) {
  ifelse(x == "scLEAD", "scLEAP", x)
}

# ---------- Classification ----------
class_df <- read_csv(
  file.path(out_dir, "runtime_classification.csv"),
  show_col_types = FALSE
)

class_order <- c(
  "scLEAD",
  "scTab",
  "Random Forest",
  "CellFM",
  "scFoundation",
  "LightGBM"
)

class_sum <- summarise_method(class_df) %>%
  mutate(method = factor(method, levels = class_order)) %>%
  arrange(method)

# ---------- Clustering ----------
clust_df <- read_csv(
  file.path(out_dir, "runtime_clustering.csv"),
  show_col_types = FALSE
)

clust_order <- c(
  "scLEAD",
  "SCANPY",
  "Seurat",
  "Pegasus",
  "Monocle3",
  "scVI",
  "SHARP"
)

clust_sum <- summarise_method(clust_df) %>%
  mutate(method = factor(method, levels = clust_order)) %>%
  arrange(method)

# ---------- LaTeX emission ----------
# Parameter counts (deep learning models assume 20k input genes and 10 output classes).

class_params <- c(
  "scLEAD"        = "142M (108M text + 34M cell)",
  "scTab"         = "9.6M",
  "Random Forest" = "100 trees",
  "CellFM"        = "86M",
  "scFoundation"  = "100M",
  "LightGBM"      = "100 trees"
)

clust_params <- c(
  "scLEAD"   = "\\texttt{<fill>}",
  "SCANPY"   = "--",
  "Seurat"   = "--",
  "Pegasus"  = "--",
  "Monocle3" = "--",
  "scVI"     = "\\texttt{<fill>}",
  "SHARP"    = "--"
)

emit_class_rows <- function(df) {

  # Ratio of each method's total time to scLEAD's total.
  # Keep "scLEAD" internally because that is the name in the CSV.
  sclead_total <- df$total_h[
    as.character(df$method) == "scLEAD"
  ][1]

  df$ratio <- df$total_h / sclead_total

  # Rename only for display in the output table.
  method_display <- display_method(as.character(df$method))

  paste(
    sprintf(
      "%-14s & %-28s & %5s & %4.1f \\\\",
      method_display,
      class_params[as.character(df$method)],
      format(
        round(df$total_h, 0),
        big.mark = ",",
        trim = TRUE
      ),
      df$ratio
    ),
    collapse = "\n"
  )
}

emit_clust_rows <- function(df) {

  # Rename only for display in the output table.
  method_display <- display_method(as.character(df$method))

  paste(
    sprintf(
      "%-10s & %3d & %-11s & %-24s & %s \\\\",
      method_display,
      df$n,
      df$median_fmt,
      df$iqr_fmt,
      clust_params[as.character(df$method)]
    ),
    collapse = "\n"
  )
}

class_tex <- paste(
  "\\begin{table}[H]",
  "\\centering",
  "\\caption{Total wall-clock annotation time across the 294-tissue atlas and model size for each within-tissue classification method.",
  "Total atlas time is the sum of end-to-end per-tissue runtimes (data processing, training, and inference) on a single NVIDIA A6000 GPU.",
  "$\\times$ scLEAP is the ratio of each method's total atlas time to scLEAP's.",
  "Parameter counts for deep learning models assume 20{,}000 input genes and 10 output classes; exact values vary slightly with gene set size and class count.}",
  "\\label{supp:runtime-classification}",
  "\\small",
  "\\begin{tabular}{l r r r}",
  "\\toprule",
  "\\textbf{Method} & \\textbf{Model size} & \\textbf{Total atlas time (h)} & \\textbf{$\\times$ scLEAP} \\\\",
  "\\midrule",
  emit_class_rows(class_sum),
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}",
  sep = "\n"
)

clust_tex <- paste(
  "\\begin{table}[H]",
  "\\centering",
  "\\caption{Runtime and model size for clustering methods.",
  "Median end-to-end wall-clock time per tissue across $n$ tissues, with interquartile range.",
  "Entries marked ``--'' are pipeline-based methods without trainable parameters.}",
  "\\label{supp:runtime-clustering}",
  "\\small",
  "\\begin{tabular}{l r l l r}",
  "\\toprule",
  "\\textbf{Method} & \\textbf{$n$} & \\textbf{Median} & \\textbf{IQR ($Q_{25}$ -- $Q_{75}$)} & \\textbf{Parameters} \\\\",
  "\\midrule",
  emit_clust_rows(clust_sum),
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}",
  sep = "\n"
)

out_tex <- file.path(
  out_dir,
  "runtime_tables.tex"
)

writeLines(
  c(
    class_tex,
    "",
    clust_tex
  ),
  out_tex
)

cat("=== Classification table ===\n")
cat(class_tex, "\n\n")

cat("=== Clustering table ===\n")
cat(clust_tex, "\n\n")

cat("Saved to:", out_tex, "\n")

