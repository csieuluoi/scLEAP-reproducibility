#!/usr/bin/env Rscript
# Figure 1 Panel D: Side-by-side bar charts (Mean F1, Precision, Recall by cell type frequency)
# Outputs: figure/bars_classification_by_frequency.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(patchwork)
})

# Paths
args <- commandArgs(trailingOnly = FALSE)

script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))

project_root <- "./"

data_root <- file.path(project_root, "resources", "scLEAP", "results", "tables")
out_dir <- file.path("figure")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Color palette matching Panel C / original figure
# Keep internal datasource/method name as scLEAD
cls_palette <- c(
  "scLEAD" = "#E64B35FF",
  "scTab" = "#4DBBD5FF",
  "CellFM" = "#00A087FF",
  "scFoundation" = "#3C5488FF",
  "Random Forest" = "#F39B7FFF",
  "lightGBM" = "#8491B4FF"
)

target_methods <- c(
  "scLEAD",
  "scTab",
  "CellFM",
  "scFoundation",
  "Random Forest",
  "lightGBM"
)

# Display names used only in the figure
method_display_labels <- c(
  "scLEAD" = "scLEAP",
  "scTab" = "scTab",
  "CellFM" = "CellFM",
  "scFoundation" = "scFdn",
  "Random Forest" = "RF",
  "lightGBM" = "lightGBM"
)

cell_order <- c("<1%", "1-5%", "5-10%", ">10%")
metrics_mean <- c("mean_f1", "mean_precision", "mean_recall")

safe_mean <- function(x) {
  if (all(is.na(x))) NA_real_ else mean(x, na.rm = TRUE)
}

# Load and filter data
df <- readr::read_csv(
  file.path(data_root, "donor_celltype_bins_ALL_METHODS.csv"),
  show_col_types = FALSE
)

# Map scLMCT -> scLEAD if present
# Keep this internal value as scLEAD
df$method[df$method == "scLMCT"] <- "scLEAD"

df <- df %>%
  filter(method %in% target_methods) %>%
  mutate(
    method = factor(method, levels = target_methods),
    cell_proportion = factor(cell_proportion, levels = cell_order)
  ) %>%
  mutate(
    across(
      all_of(metrics_mean),
      ~ suppressWarnings(as.numeric(.x))
    )
  ) %>%
  filter(
    rowSums(is.na(across(all_of(metrics_mean)))) <
      length(metrics_mean)
  )

# Aggregate: tissue-level mean, then across tissues
df_tissue <- df %>%
  group_by(method, tissue, cell_proportion) %>%
  summarise(
    across(all_of(metrics_mean), safe_mean),
    .groups = "drop"
  )

safe_sd <- function(x) {
  x2 <- x[!is.na(x)]
  if (length(x2) <= 1) 0 else sd(x2)
}

df_summary <- df_tissue %>%
  group_by(method, cell_proportion) %>%
  summarise(
    across(
      all_of(metrics_mean),
      list(
        mean = safe_mean,
        sd = safe_sd
      ),
      .names = "{.col}__{.fn}"
    ),
    .groups = "drop"
  )

df_long <- df_summary %>%
  pivot_longer(
    cols = matches("^mean_(f1|precision|recall)__"),
    names_to = c("metric_raw", ".value"),
    names_pattern = "^(mean_[^_]+)__(mean|sd)$"
  ) %>%
  mutate(
    metric = case_when(
      grepl("f1", metric_raw) ~ "Macro F1",
      grepl("precision", metric_raw) ~ "Precision",
      grepl("recall", metric_raw) ~ "Recall"
    ),
    metric = factor(
      metric,
      levels = c("Macro F1", "Precision", "Recall")
    ),
    ymin = pmax(0, mean - sd),
    ymax = pmin(1, mean + sd)
  )

# Helper: one bar panel
make_bar_panel <- function(df_sub, ylabel, show_legend = FALSE) {

  p <- ggplot(
    df_sub,
    aes(
      x = cell_proportion,
      y = mean,
      fill = method
    )
  ) +
    geom_col(
      position = position_dodge(width = 0.8),
      width = 0.7,
      na.rm = TRUE
    ) +
    geom_errorbar(
      aes(
        ymin = ymin,
        ymax = ymax
      ),
      position = position_dodge(width = 0.8),
      width = 0.25,
      linewidth = 0.3,
      na.rm = TRUE
    ) +
    scale_fill_manual(
      values = cls_palette,
      labels = method_display_labels
    ) +
    scale_y_continuous(
      breaks = seq(0, 1, 0.2),
      limits = c(0, 1),
      expand = expansion(mult = c(0, 0.02))
    ) +
    labs(
      y = ylabel,
      x = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(
      axis.text.x = element_text(size = 10),
      axis.text.y = element_text(size = 11),
      axis.title.x = element_text(size = 10),
      axis.title.y = element_text(
        size = 11,
        face = "bold"
      ),
      axis.ticks = element_line(linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.line = element_line(
        colour = "grey50",
        linewidth = 0.3
      ),
      plot.margin = margin(2, 4, 2, 2)
    )

  if (show_legend) {
    p <- p +
      theme(
        legend.position = "bottom",
        legend.title = element_blank(),
        legend.text = element_text(size = 11),
        legend.key.size = unit(0.3, "cm"),
        legend.margin = margin(0, 0, 0, 0)
      )
  } else {
    p <- p +
      theme(
        legend.position = "none"
      )
  }

  return(p)
}

p_f1 <- make_bar_panel(
  df_long %>% filter(metric == "Macro F1"),
  "Macro F1"
)

p_prec <- make_bar_panel(
  df_long %>% filter(metric == "Precision"),
  "Precision"
)

p_rec <- make_bar_panel(
  df_long %>% filter(metric == "Recall"),
  "Recall"
)

combined <- (p_f1 | p_prec | p_rec) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Classification Metrics by Cell Type Frequency",
    theme = theme(
      plot.title = element_text(
        size = 11,
        face = "bold",
        hjust = 0.5,
        margin = margin(b = 2)
      )
    )
  ) &
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.title = element_blank(),
    legend.text = element_text(size = 11),
    legend.key.height = unit(0.25, "cm"),
    legend.key.width = unit(0.6, "cm"),
    legend.margin = margin(t = -3),
    legend.box.margin = margin(t = -5)
  ) &
  guides(
    fill = guide_legend(nrow = 1)
  )

tmp_file <- file.path(
  out_dir,
  "bars_classification_by_frequency.tmp.pdf"
)

out_file <- file.path(
  out_dir,
  "bars_classification_by_frequency.pdf"
)

ggsave(
  tmp_file,
  combined,
  width = 9,
  height = 2.17,
  device = "pdf"
)

system2(
  "pdfcrop",
  c(tmp_file, out_file),
  stdout = FALSE,
  stderr = FALSE
)

file.remove(tmp_file)

message(
  "Saved: ",
  out_file
)