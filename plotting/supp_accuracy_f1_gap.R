
#!/usr/bin/env Rscript
# Supplementary dumbbell plot: per-method gap between median accuracy and median macro F1.
# A smaller gap indicates more balanced performance across common and rare cell types.
# Output: figure/supp_accuracy_f1_gap.pdf

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = FALSE)

script_path <- normalizePath(
  sub("--file=", "", args[grep("--file=", args)])
)


out_dir <- file.path("./figure")

npg_colors <- c(
  "#E64B35FF",
  "#4DBBD5FF",
  "#00A087FF",
  "#3C5488FF",
  "#F39B7FFF",
  "#8491B4FF"
)

met <- read_csv(
  "./outputs/tables/per_tissue_full_metrics.csv",
  show_col_types = FALSE
)

# ============================================================
# DISPLAY LABELS
#
# Keep "scLEAD" internally because that is the value stored
# in the input CSV, but display it as "scLEAP" in the figure.
# ============================================================

method_display <- c(
  "scLEAD"        = "scLEAP",
  "scTab"         = "scTab",
  "CellFM"        = "CellFM",
  "scFoundation"  = "scFoundation",
  "Random Forest" = "Random Forest",
  "lightGBM"      = "LightGBM"
)

# ============================================================
# SUMMARISE METHODS
# ============================================================

summary_df <- met %>%
  filter(method %in% names(method_display)) %>%
  group_by(method) %>%
  summarise(
    median_accuracy = median(accuracy, na.rm = TRUE),
    median_f1       = median(f1, na.rm = TRUE),
    gap             = median_accuracy - median_f1,
    .groups = "drop"
  ) %>%
  mutate(
    method_label = method_display[method]
  ) %>%
  arrange(gap) %>%
  mutate(
    method_label = factor(
      method_label,
      levels = method_label
    )
  )

# ============================================================
# LONG FORM
# Two points per method: Accuracy and Macro F1
# ============================================================

long_df <- summary_df %>%
  pivot_longer(
    cols = c(
      median_accuracy,
      median_f1
    ),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = factor(
      ifelse(
        metric == "median_accuracy",
        "Accuracy",
        "Macro F1"
      ),
      levels = c(
        "Macro F1",
        "Accuracy"
      )
    )
  )

# ============================================================
# PLOT
# ============================================================

p <- ggplot() +

  # Connecting line showing the gap
  geom_segment(
    data = summary_df,
    aes(
      x = median_f1,
      xend = median_accuracy,
      y = method_label,
      yend = method_label
    ),
    color = "grey60",
    linewidth = 0.8
  ) +

  # Two endpoints per method
  geom_point(
    data = long_df,
    aes(
      x = value,
      y = method_label,
      color = metric
    ),
    size = 3
  ) +

  # Gap label to the right of the accuracy endpoint
  geom_text(
    data = summary_df,
    aes(
      x = median_accuracy + 0.02,
      y = method_label,
      label = sprintf(
        "gap = %.2f",
        gap
      )
    ),
    hjust = 0,
    size = 3.2,
    color = "grey30"
  ) +

  scale_color_manual(
    values = c(
      "Macro F1" = npg_colors[1],
      "Accuracy" = npg_colors[4]
    ),
    name = NULL
  ) +

  scale_x_continuous(
    limits = c(0, 1.20),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0)
  ) +

  labs(
    x = "Median value across 294 tissues",
    y = NULL
  ) +

  theme_bw(
    base_size = 10
  ) +

  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),

    legend.position = "top",
    legend.margin = margin(
      0, 0, 0, 0
    ),

    axis.text.y = element_text(
      size = 10
    )
  )

# ============================================================
# SAVE
# ============================================================

out_file <- file.path(
  out_dir,
  "supp_accuracy_f1_gap.pdf"
)

ggsave(
  out_file,
  p,
  width = 6.5,
  height = 3.0,
  device = cairo_pdf
)

cat(
  "Wrote:",
  out_file,
  "\n"
)

# ============================================================
# MANUSCRIPT AUDIT SUMMARY
# ============================================================

cat(
  "\nMedian accuracy and F1 per method (sorted by gap ascending):\n"
)

print(
  summary_df %>%
    select(
      method,
      method_label,
      median_accuracy,
      median_f1,
      gap
    ),
  n = 6
)
