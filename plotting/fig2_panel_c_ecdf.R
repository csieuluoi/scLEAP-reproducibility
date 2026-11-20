#!/usr/bin/env Rscript
# Figure 2 Panel C: Empirical CDF of per-cell-type one-vs-rest F1 for each
# method. Reads the per-CT bootstrap CSV produced by
# `plotting/tables/compute_zeroshot_bootstrap.py` (49 evaluated cell types,
# after parent-collapse + antichain filtering) and draws one step-curve per
# method.
#
# Output: figure/zeroshot_ecdf.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

args <- commandArgs(trailingOnly = FALSE)

project_root <- "./"
boot_dir <- file.path(project_root, "outputs", "tables")
out_dir <- file.path("figure")

FIG_WIDTH  <- 1.6 * 3
FIG_HEIGHT <- 1.6 * 1.9

BASE_FONT_SIZE   <- 12 - 1
AXIS_TEXT_SIZE   <- 10.5 - 0.2
AXIS_TITLE_SIZE  <- 12 - 1
LEGEND_TEXT_SIZE <- 12 - 1

LINE_WIDTH <- 0.85

PLOT_MARGIN <- margin(5, 10, 5, 5)

# Keep internal datasource/method names unchanged
method_colors <- c(
  "scLEAD"        = "#E64B35FF",
  "LangCell"      = "#4DBBD5FF",
  "CellWhisperer" = "#00A087FF"
)

method_levels <- names(method_colors)

# Display names used only in the figure
method_display_labels <- c(
  "scLEAD"        = "scLEAP",
  "LangCell"      = "LangCell",
  "CellWhisperer" = "CellWhisperer"
)

boot_fine <- read_csv(
  file.path(
    boot_dir,
    "zeroshot_per_celltype_bootstrap.csv"
  ),
  show_col_types = FALSE
) %>%
  mutate(
    method = factor(
      method,
      levels = method_levels
    )
  )

ecdf_df <- boot_fine %>%
  group_by(method) %>%
  arrange(
    desc(F1_observed),
    .by_group = TRUE
  ) %>%
  mutate(
    frac = seq_len(n()) / n()
  ) %>%
  ungroup()

thresholds <- c(0.5, 0.8)

ref_lines <- tibble::tibble(
  t = thresholds
)

shared_theme <- theme_minimal(
  base_size = BASE_FONT_SIZE
) +
  theme(
    axis.line = element_line(
      colour = "grey50",
      linewidth = 0.35
    ),

    panel.grid.minor = element_blank(),

    panel.grid.major = element_line(
      colour = "grey92",
      linewidth = 0.25
    ),

    axis.text = element_text(
      size = AXIS_TEXT_SIZE
    ),

    axis.title = element_text(
      size = AXIS_TITLE_SIZE
    ),

    axis.title.x = element_text(
      size = AXIS_TITLE_SIZE,
      margin = margin(t = 8)
    ),

    legend.position = c(1, 1),
    legend.justification = c(1, 1),
    legend.direction = "vertical",

    legend.background = element_rect(
      fill = scales::alpha("white", 0.85),
      colour = NA
    ),

    legend.key = element_rect(
      fill = NA,
      colour = NA
    ),

    legend.key.height = unit(
      0.35,
      "cm"
    ),

    legend.key.width = unit(
      0.65,
      "cm"
    ),

    legend.text = element_text(
      size = LEGEND_TEXT_SIZE
    ),

    legend.title = element_blank(),

    legend.margin = margin(
      2,
      4,
      2,
      4
    ),

    plot.margin = PLOT_MARGIN
  )

p_ecdf <- ggplot(
  ecdf_df,
  aes(
    x = frac,
    y = F1_observed,
    colour = method
  )
) +
  geom_hline(
    data = ref_lines,
    aes(
      yintercept = t
    ),
    linetype = "dashed",
    colour = "grey25",
    linewidth = 0.55
  ) +
  geom_step(
    direction = "vh",
    linewidth = LINE_WIDTH
  ) +
  geom_text(
    data = ref_lines,
    aes(
      x = 1.02,
      y = t,
      label = sprintf(
        "t = %.1f",
        t
      )
    ),
    inherit.aes = FALSE,
    hjust = 1,
    vjust = 1.2,
    size = AXIS_TEXT_SIZE / ggplot2::.pt,
    colour = "grey25"
  ) +
  scale_colour_manual(
    values = method_colors,
    labels = method_display_labels,
    drop = FALSE,
    name = NULL
  ) +
  scale_x_continuous(
    limits = c(0, 1.04),
    breaks = seq(0, 1, 0.2),
    labels = scales::percent_format(
      accuracy = 1
    ),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0)
  ) +
  labs(
    title = NULL,
    x = "Fraction of cell types with F1 >= t",
    y = "Per-cell-type F1 threshold (t)"
  ) +
  shared_theme

tmp_file <- file.path(
  out_dir,
  "zeroshot_ecdf.tmp.pdf"
)

ggsave(
  tmp_file,
  p_ecdf,
  width = FIG_WIDTH,
  height = FIG_HEIGHT,
  device = "pdf"
)

message(
  "Saved: ",
  basename(tmp_file)
)

cov <- boot_fine %>%
  group_by(method) %>%
  summarise(
    n_total = n(),
    ge_0.5 = sum(F1_observed >= 0.5),
    ge_0.8 = sum(F1_observed >= 0.8),
    .groups = "drop"
  )

print(cov)

out_file <- file.path(
  out_dir,
  "zeroshot_ecdf.pdf"
)

message(
  "out_dir = ",
  out_dir
)

message(
  "tmp_file = ",
  tmp_file
)

message(
  "out_file = ",
  out_file
)

system2(
  "pdfcrop",
  c(
    "--margins='0 0 0 0'",
    tmp_file,
    out_file
  ),
  stdout = FALSE,
  stderr = FALSE
)

file.remove(tmp_file)

message(
  "Saved: zeroshot_ecdf.pdf"
)