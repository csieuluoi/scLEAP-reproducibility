#!/usr/bin/env Rscript
# Generate dataset inventory LaTeX tables for supplementary.
# Reads metadata CSVs from dung-code/ (read-only) and per_tissue_full_metrics.csv.
# Outputs: outputs/tables/dataset_tables.tex
#          supplementary-data/Supplementary_Data_S1_atlas_inventory.csv

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
project_root <- "."
meta_dir <- file.path(project_root, "scLEAD_data_processing", "meta-data")
out_dir <- file.path(project_root, "outputs", "tables")
suppdata_dir <- file.path(project_root, "supplementary-data")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(suppdata_dir, showWarnings = FALSE, recursive = TRUE)

# ============================================================
# 1. Zero-shot external datasets table (10 rows, inline LaTeX)
# ============================================================
zs <- read_csv(file.path(out_dir, "zeroshot_dataset_summary_v2.csv"), show_col_types = FALSE)

escape_tex <- function(x) {
  x <- gsub("&", "\\\\&", x)
  x <- gsub("_", "\\\\_", x)
  x <- gsub("%", "\\\\%", x)
  x <- gsub("#", "\\\\#", x)
  x
}

fmt_num <- function(x) formatC(x, format = "d", big.mark = ",")

zs_rows <- zs %>%
  arrange(Tissue, `Dataset ID`) %>%
  mutate(
    ct_list = gsub(";", ",", cell_types),
    row_tex = sprintf(
      "%s & %s & %s & %s & %d & %s",
      escape_tex(`Dataset ID`),
      escape_tex(Tissue),
      fmt_num(n_cells),
      fmt_num(n_genes),
      n_cell_types,
      escape_tex(ct_list)
    )
  )

zs_tex <- paste(
  "\\begin{table}[H]",
  "\\centering",
  "\\caption{External datasets for zero-shot annotation evaluation.",
  "Each dataset was preprocessed using the same normalization pipeline as the training data.",
  "None of these datasets are included in the CELLxGENE Census database.}",
  "\\label{supp:zeroshot-datasets}",
  "\\small",
  "\\begin{tabular}{l l r r r p{8cm}}",
  "\\toprule",
  "\\textbf{Dataset} & \\textbf{Tissue} & \\textbf{Cells} & \\textbf{Genes} & \\textbf{Types} & \\textbf{Cell Types} \\\\",
  "\\midrule",
  paste(zs_rows$row_tex, collapse = " \\\\\n"),
  "\\\\",
  "\\bottomrule",
  "\\end{tabular}",
  "\\end{table}",
  sep = "\n"
)

# ============================================================
# 2. Atlas per-tissue summary table (294 rows, 2 tissues per row)
# ============================================================
metrics <- read_csv(file.path(out_dir, "per_tissue_full_metrics.csv"), show_col_types = FALSE)

tissue_summary <- metrics %>%
  filter(method == "scLEAD") %>%
  select(tissue, n_cell_types, n_test_cells) %>%
  arrange(tissue) %>%
  mutate(
    tissue_fmt = gsub("_", " ", tissue),
    tissue_fmt = tools::toTitleCase(tissue_fmt)
  )

n_tissues <- nrow(tissue_summary)
half <- ceiling(n_tissues / 2)

left <- tissue_summary[1:half, ]
right_idx <- (half + 1):n_tissues
right <- tissue_summary[right_idx, ]
if (nrow(right) < nrow(left)) {
  right <- bind_rows(right, tibble(tissue = NA, n_cell_types = NA, n_test_cells = NA,
                                    tissue_fmt = ""))
}

atlas_body_rows <- mapply(function(i) {
  l <- left[i, ]
  r <- right[i, ]
  r_tissue <- if (is.na(r$tissue)) "" else escape_tex(r$tissue_fmt)
  r_ct <- if (is.na(r$n_cell_types)) "" else as.character(r$n_cell_types)
  r_cells <- if (is.na(r$n_test_cells)) "" else fmt_num(r$n_test_cells)
  sprintf("%s & %d & %s & %s & %s & %s",
          escape_tex(l$tissue_fmt), l$n_cell_types, fmt_num(l$n_test_cells),
          r_tissue, r_ct, r_cells)
}, 1:nrow(left))

atlas_tex <- paste(
  "\\begin{longtable}{l r r l r r}",
  "\\caption{Per-tissue summary of the CELLxGENE atlas used for within-tissue classification.",
  sprintf("The atlas contains %d tissues, %s total test cells, and %d to %d cell types per tissue.}",
          n_tissues, fmt_num(sum(tissue_summary$n_test_cells)),
          min(tissue_summary$n_cell_types), max(tissue_summary$n_cell_types)),
  "\\label{supp:atlas-tissues-detail} \\\\",
  "\\toprule",
  "\\textbf{Tissue} & \\textbf{Types} & \\textbf{Test Cells} & \\textbf{Tissue} & \\textbf{Types} & \\textbf{Test Cells} \\\\",
  "\\midrule",
  "\\endfirsthead",
  "\\toprule",
  "\\textbf{Tissue} & \\textbf{Types} & \\textbf{Test Cells} & \\textbf{Tissue} & \\textbf{Types} & \\textbf{Test Cells} \\\\",
  "\\midrule",
  "\\endhead",
  "\\midrule",
  "\\multicolumn{6}{r}{\\textit{Continued on next page}} \\\\",
  "\\endfoot",
  "\\bottomrule",
  "\\endlastfoot",
  paste(atlas_body_rows, collapse = " \\\\\n"),
  "\\\\",
  "\\end{longtable}",
  sep = "\n"
)

# ============================================================
# 3. Copy full atlas CSV as Supplementary Data S1
# ============================================================
file.copy(
  file.path(meta_dir, "cellxgene_dataset_summary_human_2025-01-30_all.csv"),
  file.path(suppdata_dir, "Supplementary_Data_S1_atlas_inventory.csv"),
  overwrite = TRUE
)

# ============================================================
# Write combined LaTeX output
# ============================================================
out_tex <- file.path(out_dir, "dataset_tables.tex")
writeLines(c(
  "% === Zero-shot external datasets ===",
  zs_tex,
  "",
  "% === Atlas per-tissue summary (2 tissues per row) ===",
  atlas_tex
), out_tex)

cat("Saved:", out_tex, "\n")
cat("Saved:", file.path(suppdata_dir, "Supplementary_Data_S1_atlas_inventory.csv"), "\n")
cat(sprintf("Zero-shot: %d datasets\n", nrow(zs)))
cat(sprintf("Atlas: %d tissues\n", n_tissues))
