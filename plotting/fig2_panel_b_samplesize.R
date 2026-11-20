#!/usr/bin/env Rscript

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

boot_dir <- file.path(project_root, "outputs", "tables")
out_dir <- file.path("figure")

bootstrap_size_csv <- file.path(boot_dir, "zeroshot_samplesize_bootstrap.csv")

# Final combined-PDF dimensions in inches. Adjust these two values to make
# the three-panel row wider/narrower or taller/shorter when rerunning.
FIG_WIDTH  <- 1.2 * 9
FIG_HEIGHT <- 1.2 * 1.9

BASE_FONT_SIZE   <- 12
AXIS_TEXT_SIZE   <- 11
AXIS_TITLE_SIZE  <- 12
LEGEND_TEXT_SIZE <- 12

LINE_WIDTH <- 0.85
POINT_SIZE <- 2.4

PLOT_MARGIN <- margin(5, 10, 5, 5)

# Keep internal method names unchanged
method_colors <- c(
  "scLEAD"        = "#E64B35FF",
  "LangCell"      = "#4DBBD5FF",
  "CellWhisperer" = "#00A087FF"
)

method_levels <- names(method_colors)

# Display labels used only in the figure
method_display_labels <- c(
  "scLEAD"        = "scLEAP",
  "LangCell"      = "LangCell",
  "CellWhisperer" = "CellWhisperer"
)

size_order <- c("50", "100", "200", "500", "1000", "2000", ">2000")

boot_sz <- read_csv(
  bootstrap_size_csv,
  show_col_types = FALSE
)

pivot_metrics <- function(df) {
  df %>%
    pivot_longer(
      cols = matches("^(F1|Precision|Recall)_(observed|ci_lo|ci_hi)$"),
      names_to = c("metric", ".value"),
      names_pattern = "^(F1|Precision|Recall)_(observed|ci_lo|ci_hi)$"
    ) %>%
    rename(
      value = observed,
      lo = ci_lo,
      hi = ci_hi
    ) %>%
    mutate(
      metric = factor(
        metric,
        levels = c("F1", "Precision", "Recall")
      ),
      method = factor(
        method,
        levels = method_levels
      ),
      ymin = pmax(0, lo),
      ymax = pmin(1, hi)
    )
}

df_size <- boot_sz %>%
  pivot_metrics() %>%
  mutate(
    size_bin = recode(
      size_bin,
      "[1-50]"      = "50",
      "[51-100]"    = "100",
      "[101-200]"   = "200",
      "[101-500]"   = "500",
      "[201-500]"   = "500",
      "[501-1000]"  = "1000",
      "[1001-2000]" = "2000",
      "[>2000]"     = ">2000"
    ),
    size_bin = factor(
      size_bin,
      levels = size_order
    ),
    x_num = as.integer(size_bin)
  )

shared_theme <- theme_minimal(base_size = BASE_FONT_SIZE) +
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

    # Legend: bottom-right
    legend.position = "inside",
    legend.position.inside = c(0.98, 0.02),
    legend.justification = c(1, 0),

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

make_size_plot <- function(
  metric_name,
  y_label,
  show_x_title = FALSE,
  show_legend = FALSE
) {

  p <- df_size %>%
    filter(
      metric == metric_name
    ) %>%
    ggplot(
      aes(
        x = x_num,
        y = value,
        colour = method,
        group = method
      )
    ) +
    geom_errorbar(
      aes(
        ymin = ymin,
        ymax = ymax
      ),
      width = 0.12,
      linewidth = 0.35
    ) +
    geom_line(
      linewidth = LINE_WIDTH
    ) +
    geom_point(
      size = POINT_SIZE
    ) +
    scale_colour_manual(
      values = method_colors,
      labels = method_display_labels,
      drop = FALSE,
      name = NULL
    ) +
    scale_x_continuous(
      breaks = seq_along(size_order),
      labels = size_order,
      limits = c(
        0.6,
        length(size_order) + 0.4
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
      x = if (show_x_title) {
        "Cell-type abundance group"
      } else {
        NULL
      },
      y = y_label
    ) +
    shared_theme +
    guides(
      colour = guide_legend(
        ncol = 1,
        byrow = TRUE,
        override.aes = list(
          linewidth = 1.2,
          size = 3
        )
      )
    )

  if (!show_legend) {
    p <- p +
      theme(
        legend.position = "none"
      )
  }

  p
}

plots <- list(
  make_size_plot(
    "F1",
    "Macro F1"
  ),
  make_size_plot(
    "Precision",
    "Precision",
    show_x_title = TRUE
  ),
  make_size_plot(
    "Recall",
    "Recall",
    show_legend = TRUE
  )
)

p <- Reduce(`|`, plots) +
  plot_layout(
    widths = rep(
      1,
      length(plots)
    )
  )

tmp_file <- file.path(
  out_dir,
  "zeroshot_merged_metrics_samplesize.tmp.pdf"
)

out_file <- file.path(
  out_dir,
  "zeroshot_merged_metrics_samplesize.pdf"
)

ggsave(
  tmp_file,
  p,
  width = FIG_WIDTH,
  height = FIG_HEIGHT,
  device = "pdf"
)

# Crop white margins for final figure assembly. Keep the temporary PDF if
# pdfcrop fails so the uncropped output remains available for diagnosis.
crop_status <- system2(
  "pdfcrop",
  c(
    "--margins='0 0 0 0'",
    tmp_file,
    out_file
  ),
  stdout = FALSE,
  stderr = FALSE
)

if (crop_status != 0) {
  stop(
    "pdfcrop failed; uncropped output retained at: ",
    tmp_file
  )
}

file.remove(tmp_file)

message(
  "Saved: ",
  out_file
)