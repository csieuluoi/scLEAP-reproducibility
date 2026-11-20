#!/usr/bin/env Rscript
# Figure 1 Panel C: Stacked violin plots (F1, Precision, Recall) — matching original figure
# Outputs: figure/violin_classification_stacked.pdf

library(ggplot2)
library(dplyr)
library(patchwork)

# Paths
args <- commandArgs(trailingOnly = FALSE)

script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))

project_root <- "./"

data_root <- file.path(project_root, "resources", "scLEAP", "results", "tables")
out_dir <- file.path("figure")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Color palette matching original
# Keep the underlying method key as scLEAD (datasource/result value),
# but display it as scLEAP in the figure.
cls_palette <- c(
  "scLEAD" = "#E64B35FF",
  "scTab" = "#4DBBD5FF",
  "CellFM" = "#00A087FF",
  "scFoundation" = "#3C5488FF",
  "Random Forest" = "#F39B7FFF",
  "lightGBM" = "#8491B4FF"
)

target_methods <- c("scLEAD", "scTab", "CellFM", "scFoundation", "Random Forest", "lightGBM")

# Display labels for visualization only
method_display_labels <- c(
  "scLEAD" = "scLEAP",
  "scTab" = "scTab",
  "CellFM" = "CellFM",
  "scFoundation" = "scFdn",
  "Random Forest" = "RF",
  "lightGBM" = "lightGBM"
)

# Load and filter data
data <- read.csv(file.path(data_root, "cell_type_classification.csv"))
data <- data[data$tissue != "skeletal muscle tissue", ]
data <- data[data$method %in% target_methods, ]
data$method <- factor(data$method, levels = target_methods)

# Helper: create one violin row (no x-axis labels, compact)
make_violin_row <- function(data, metric, ylabel, show_x = FALSE) {
  p <- ggplot(data, aes(x = method, y = .data[[metric]], fill = method)) +
    geom_violin(width = 1.1, alpha = 1, color = "black", linewidth = 0.3, trim = TRUE) +
    geom_boxplot(width = 0.1, outlier.shape = NA, color = "black", fill = NA, linewidth = 0.4) +
    scale_fill_manual(
      values = cls_palette,
      labels = method_display_labels
    ) +
    scale_x_discrete(
      labels = method_display_labels,
      expand = expansion(mult = 0.02, add = 0)
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      limits = c(0, 1),
      expand = c(0, 0)
    ) +
    labs(y = ylabel, x = NULL) +
    theme_minimal(base_size = 10.5) +
    theme(
      axis.text.y = element_text(size = 9.5),
      axis.title.y = element_text(size = 10.5, face = "bold"),
      axis.ticks.y = element_line(linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.line = element_line(colour = "grey50", linewidth = 0.3),
      plot.margin = margin(5, 2, 5, 2),
      legend.position = "none"
    )

  if (show_x) {
    p <- p + theme(axis.text.x = element_text(angle = 25, hjust = 1, size = 10))
  } else {
    p <- p + theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
  }

  return(p)
}

p_f1   <- make_violin_row(data, "f1", "Macro F1", show_x = FALSE)
p_prec <- make_violin_row(data, "precision", "Precision", show_x = FALSE)
p_rec  <- make_violin_row(data, "recall", "Recall", show_x = TRUE)

combined <- p_f1 / p_prec / p_rec + plot_layout(heights = c(1, 1, 1.15))

tmp_file <- file.path(out_dir, "violin_classification_stacked.tmp.pdf")
out_file <- file.path(out_dir, "violin_classification_stacked.pdf")

ggsave(tmp_file, combined, width = 3.6, height = 3.4, device = "pdf")
system2("pdfcrop", c(tmp_file, out_file), stdout = FALSE, stderr = FALSE)

# system2(
#   "pdfcrop",
#   c("--margins='5 2 6 0'", tmp_file, out_file),
#   stdout = FALSE,
#   stderr = FALSE
# )

file.remove(tmp_file)
message("Saved: ", out_file)

# env -u TEXMFROOT -u TEXMFHOME -u TEXMFCNF -u TEXINPUTS -u PERL5LIB \
#   PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH \
#   Rscript fig1_panel_c_violin.R