#!/usr/bin/env Rscript
# Figure 3 Panel C: Clustering performance violin plots (NMI, ARI, Completeness)
# Outputs: figure/fig3_clustering_metrics.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)

script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))

project_root <- "./"

data_root <- file.path(project_root, "resources", "scLEAP", "results", "tables")
out_dir <- file.path("figure")

# Load data
df <- read_csv(file.path(data_root, "cell_type_clustering.csv"), show_col_types = FALSE)
# Rename old method labels -> scLEAP
df$method[df$method %in% c("scLMCT", "scLEAD")] <- "scLEAP"

# Exclude skeletal muscle tissue (consistent with other panels)
df <- df %>% filter(tissue != "skeletal muscle tissue")

# Method order
method_order <- c(
  "scLEAP",
  "scanpy",
  "Seurat",
  "SHARP",
  "scVI",
  "pegasus",
  "monocle3"
)

df$method <- factor(df$method, levels = method_order)

# Display labels
method_labels <- c(
  "scLEAP"   = "scLEAP",
  "scanpy"   = "SCANPY",
  "Seurat"   = "Seurat",
  "SHARP"    = "SHARP",
  "scVI"     = "scVI",
  "pegasus"  = "Pegasus",
  "monocle3" = "Monocle3"
)

# Color palette
method_colors <- c(
  "scLEAP"   = "#E64B35FF",
  "scanpy"   = "#4DBBD5FF",
  "Seurat"   = "#00A087FF",
  "SHARP"    = "#3C5488FF",
  "scVI"     = "#F39B7FFF",
  "pegasus"  = "#8491B4FF",
  "monocle3" = "#91D1C2FF"
)
make_violin <- function(data, metric, ylabel, show_x = FALSE) {
  p <- ggplot(data, aes(x = method, y = .data[[metric]], fill = method)) +
    geom_violin(width = 1.0, alpha = 1, color = "black", linewidth = 0.3) +
    geom_boxplot(width = 0.1, outlier.shape = NA, color = "black", fill = NA, linewidth = 0.4) +
    scale_fill_manual(values = method_colors) +
    scale_x_discrete(labels = method_labels) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2), expand = c(0, 0)) +
    labs(y = ylabel, x = NULL) +
    theme_minimal(base_size = 9) +
    theme(
      axis.text.y = element_text(size = 7, color = "grey30"),
      axis.title.y = element_text(size = 8, face = "bold"),
      axis.ticks.y = element_line(color = "grey40", linewidth = 0.3),
      axis.line.x = element_line(color = "grey40", linewidth = 0.4),
      axis.line.y = element_line(color = "grey40", linewidth = 0.4),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(4, 2, 4, 2),
      legend.position = "none"
    )

  if (show_x) {
    p <- p + theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8))
  } else {
    p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  }

  return(p)
}

# Use full metric names to match Panel B y-axis label widths
p_nmi <- make_violin(df, "nmi", "NMI", show_x = TRUE)
p_ari <- make_violin(df, "ari", "ARI", show_x = TRUE)
p_comp <- make_violin(df, "completeness", "Completeness", show_x = TRUE)

# Align y-axes with Panel B by collecting within each column
col1 <- p_nmi
col2 <- p_ari
col3 <- p_comp

combined <- col1 | col2 | col3

tmp_file <- file.path(out_dir, "fig3_clustering_metrics.tmp.pdf")
out_file <- file.path(out_dir, "fig3_clustering_metrics.pdf")
ggsave(tmp_file, combined, width = 7.5, height = 1.995, device = "pdf")
# system2("pdfcrop", c(tmp_file, out_file), stdout = FALSE, stderr = FALSE)
system2(
  "pdfcrop",
  c("--margins='0 0 0 0'", tmp_file, out_file),
  stdout = FALSE,
  stderr = FALSE
)
file.remove(tmp_file)

# out_file <- file.path(out_dir, "fig3_clustering_metrics.pdf")
# ggsave(out_file, combined, width = 7.5, height = 1.995, device = "pdf")
message("Saved: ", out_file)
