#!/usr/bin/env Rscript
# Supplementary runtime line plot: runtime vs dataset size (binned by cell count).
# Output: figures/supp_runtime_classification_line.pdf

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))

out_dir <- "./figure"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

npg_colors <- c(
  "#E64B35FF", "#4DBBD5FF", "#00A087FF",
  "#3C5488FF", "#F39B7FFF", "#8491B4FF"
)

rt <- read_csv(
  file.path(
    "./",
    "outputs", "tables", "runtime_classification.csv"
  ),
  show_col_types = FALSE
)

met <- read_csv(
  file.path(
    "./",
    "outputs", "tables", "per_tissue_full_metrics.csv"
  ),
  show_col_types = FALSE
)

cell_counts <- met %>%
  filter(method == "scLEAD") %>%
  select(tissue, n_test_cells)

# Keep internal names for filtering / ordering
method_order <- c(
  "scLEAD",
  "scTab",
  "CellFM",
  "scFoundation",
  "Random Forest",
  "LightGBM"
)

# Display names for the figure
method_labels <- c(
  "scLEAD"        = "scLEAP",
  "scTab"         = "scTab",
  "CellFM"        = "CellFM",
  "scFoundation"  = "scFoundation",
  "Random Forest" = "Random Forest",
  "LightGBM"      = "LightGBM"
)

display_order <- unname(method_labels[method_order])

df <- rt %>%
  filter(method %in% method_order) %>%
  inner_join(cell_counts, by = "tissue") %>%
  mutate(
    method = factor(method, levels = method_order),
    method_label = factor(
      method_labels[as.character(method)],
      levels = display_order
    ),
    total_min = total_sec / 60,
    size_bin = cut(
      n_test_cells,
      breaks = c(0, 1000, 5000, 10000, 50000, 100000, 200000, 500000, Inf),
      labels = c("<1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", "100K-200K", "200K-500K", ">500K"),
      right = TRUE
    )
  )

df_summary <- df %>%
  group_by(method, method_label, size_bin) %>%
  summarise(
    median_min = median(total_min, na.rm = TRUE),
    q25 = quantile(total_min, 0.25, na.rm = TRUE),
    q75 = quantile(total_min, 0.75, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )

p <- ggplot(
  df_summary,
  aes(x = size_bin, y = median_min, color = method_label, group = method_label)
) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.5) +
  geom_ribbon(
    aes(ymin = q25, ymax = q75, fill = method_label),
    alpha = 0.1,
    color = NA
  ) +
  scale_color_manual(
    values = setNames(npg_colors, display_order),
    breaks = display_order,
    name = "Method"
  ) +
  scale_fill_manual(
    values = setNames(npg_colors, display_order),
    breaks = display_order,
    guide = "none"
  ) +
  scale_y_log10(
    labels = function(x) {
      ifelse(
        x >= 60,
        paste0(round(x / 60, 1), " h"),
        paste0(round(x, 1), " min")
      )
    }
  ) +
  labs(
    x = "Number of test cells",
    y = "End-to-end time (log scale)"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    legend.position = "right",
    legend.text = element_text(size = 7),
    legend.title = element_text(size = 8, face = "bold"),
    legend.key.size = grid::unit(0.4, "cm"),
    axis.text.x = element_text(angle = 30, hjust = 1, size = 7),
    panel.grid.minor = element_blank()
  )

out_file <- file.path(out_dir, "supp_runtime_classification_line.pdf")

ggsave(
  out_file,
  p,
  width = 5.5,
  height = 3.5,
  device = "pdf"
)

message("Saved: ", out_file)