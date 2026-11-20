#!/usr/bin/env Rscript

# Run from this plotting directory with:
# conda run -n R-PWA Rscript fig2_panel_b_zeroshot_metrics.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(patchwork)
  library(grid)
})

project_root <- "./"
data_dir <- file.path(project_root, "outputs", "tables")
out_dir <- "figure"

bootstrap_csv <- file.path(data_dir, "zeroshot_per_dataset_bootstrap.csv")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# =========================================================
# EXPORT / PANEL SIZE CONTROLS
# =========================================================

# Adjust the shared-x-ticks figure here. The separate-x-ticks figure uses the
# same dimensions by default so the two versions have matching proportions.
# pdfcrop trims both final pages after they are saved.
SHARED_FIG_WIDTH_IN       <- 5
SHARED_HEIGHT_TO_WIDTH    <- 1.3
SHARED_FIG_HEIGHT_IN      <- SHARED_FIG_WIDTH_IN * SHARED_HEIGHT_TO_WIDTH

METRIC_LEVELS <- c("F1", "Precision", "Recall")

METRIC_LABELS <- c(
  "F1" = "Macro F1",
  "Precision" = "Precision",
  "Recall" = "Recall"
)

SEPARATE_FIG_WIDTH_IN  <- SHARED_FIG_WIDTH_IN
SEPARATE_FIG_HEIGHT_IN <- SHARED_FIG_HEIGHT_IN

# =========================================================
# FONT CONTROLS
# =========================================================

BASE_FONT_SIZE      <- 12
AXIS_TEXT_SIZE      <- 10.5
AXIS_TITLE_SIZE     <- 12
STRIP_TEXT_SIZE     <- 12
LEGEND_TEXT_SIZE    <- 12
DATASET_X_TEXT_SIZE <- 10.5

# Compact type used by both narrow three-row figures.
COMPACT_AXIS_TEXT_SIZE      <- 7.5
COMPACT_AXIS_TITLE_SIZE     <- 9.5
COMPACT_STRIP_TEXT_SIZE     <- 8
COMPACT_LEGEND_TEXT_SIZE    <- 8
COMPACT_DATASET_X_TEXT_SIZE <- 7.5

# =========================================================
# GEOM CONTROLS
# =========================================================

DATASET_BAR_WIDTH  <- 0.7
DODGE_WIDTH        <- 0.8
ERRORBAR_WIDTH     <- 0.25
ERRORBAR_LINEWIDTH <- 0.3

# Narrower grouped bars used by both three-row figures.
COMPACT_DATASET_BAR_WIDTH <- 0.56
COMPACT_DODGE_WIDTH       <- 0.68
COMPACT_ERRORBAR_WIDTH    <- 0.18

# =========================================================
# LEGEND CONTROLS
# =========================================================

LEGEND_KEY_HEIGHT <- 0.35
LEGEND_KEY_WIDTH  <- 0.65

# =========================================================
# SPACING / MARGIN CONTROLS
# =========================================================

PLOT_MARGIN_TOP    <- 6
PLOT_MARGIN_RIGHT  <- 6
PLOT_MARGIN_BOTTOM <- 6
PLOT_MARGIN_LEFT   <- 6

FACET_SPACING_X <- 0.22

# Independent vertical spacing controls for the compact three-row layouts.
METRIC_PLOT_GAP_PT      <- 12
BOTTOM_TO_LEGEND_GAP_PT <- 0

# =========================================================
# COLORS / ORDERING
# =========================================================

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

tissue_map <- c(
  "E-MTAB-9221" = "Blood",
  "GSE185948"   = "Kidn.",
  "E-HCAD-9"    = "Liver",
  "E-CURD-114"  = "Lung",
  "E-HCAD-15"   = "Lung",
  "E-MTAB-5061" = "Pancreas",
  "GSE81608"    = "Pancreas",
  "GSE84133"    = "Pancreas",
  "GSE86473"    = "Pancreas",
  "GSE111976"   = "Endo."
)

# =========================================================
# LOAD / PREPARE DATA
# =========================================================

boot_ds <- read_csv(
  bootstrap_csv,
  show_col_types = FALSE
)

# Use one common dataset order for all three metrics so the stacked plots are
# directly comparable. The order follows the original F1/scLEAD arrangement.
dataset_order <- boot_ds %>%
  filter(method == "scLEAD") %>%
  mutate(tissue = tissue_map[dataset]) %>%
  arrange(tissue, desc(F1_observed)) %>%
  pull(dataset)

tissue_levels <- unique(
  unname(tissue_map[dataset_order])
)

df_dataset <- boot_ds %>%
  mutate(
    tissue = tissue_map[dataset]
  ) %>%
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
      levels = METRIC_LEVELS
    ),
    method = factor(
      method,
      levels = method_levels
    ),
    dataset = factor(
      dataset,
      levels = dataset_order
    ),
    tissue = factor(
      tissue,
      levels = tissue_levels
    ),
    ymin = pmax(0, lo),
    ymax = pmin(1, hi)
  )

# =========================================================
# THEME
# =========================================================

bar_theme <- theme_minimal(
  base_size = BASE_FONT_SIZE
) +
  theme(
    axis.line = element_line(
      colour = "grey50",
      linewidth = 0.35
    ),

    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),

    axis.text.x = element_text(
      size = DATASET_X_TEXT_SIZE,
      angle = 45,
      hjust = 0.95,
      vjust = 1,
      margin = margin(t = 2),
      lineheight = 0.9
    ),

    axis.text.y = element_text(
      size = AXIS_TEXT_SIZE
    ),

    axis.title.y = element_text(
      size = AXIS_TITLE_SIZE
    ),

    axis.title.x = element_blank(),

    strip.placement = "inside",

    strip.text.x = element_text(
      size = STRIP_TEXT_SIZE,
      face = "plain"
    ),

    strip.background = element_blank(),

    panel.spacing.x = unit(
      FACET_SPACING_X,
      "cm"
    ),

    legend.position = "bottom",
    legend.title = element_blank(),

    legend.text = element_text(
      size = LEGEND_TEXT_SIZE
    ),

    legend.key = element_rect(
      fill = NA,
      colour = NA
    ),

    legend.key.height = unit(
      LEGEND_KEY_HEIGHT,
      "cm"
    ),

    legend.key.width = unit(
      LEGEND_KEY_WIDTH,
      "cm"
    ),

    plot.margin = margin(
      PLOT_MARGIN_TOP,
      PLOT_MARGIN_RIGHT,
      PLOT_MARGIN_BOTTOM,
      PLOT_MARGIN_LEFT
    )
  )

# =========================================================
# HELPERS
# =========================================================

make_dataset_panel <- function(
  metric_name,
  show_x_labels = TRUE,
  show_tissue_labels = TRUE,
  panel_index = 1,
  panel_count = length(METRIC_LEVELS),
  compact_style = FALSE
) {

  panel_data <- df_dataset %>%
    filter(
      metric == metric_name
    )

  panel_bar_width <- if (compact_style) {
    COMPACT_DATASET_BAR_WIDTH
  } else {
    DATASET_BAR_WIDTH
  }

  panel_dodge_width <- if (compact_style) {
    COMPACT_DODGE_WIDTH
  } else {
    DODGE_WIDTH
  }

  panel_errorbar_width <- if (compact_style) {
    COMPACT_ERRORBAR_WIDTH
  } else {
    ERRORBAR_WIDTH
  }

  p <- ggplot(
    panel_data,
    aes(
      x = dataset,
      y = value,
      fill = method
    )
  ) +
    geom_col(
      position = position_dodge(
        width = panel_dodge_width
      ),
      width = panel_bar_width
    ) +
    geom_errorbar(
      aes(
        ymin = ymin,
        ymax = ymax,
        group = method
      ),
      position = position_dodge(
        width = panel_dodge_width
      ),
      width = panel_errorbar_width,
      linewidth = ERRORBAR_LINEWIDTH,
      colour = "grey30"
    ) +
    scale_fill_manual(
      values = method_colors,
      labels = method_display_labels,
      drop = FALSE,
      name = NULL
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      expand = c(0, 0)
    ) +
    facet_grid(
      ~ tissue,
      scales = "free_x",
      space = "free_x"
    ) +
    labs(
      y = unname(METRIC_LABELS[metric_name]),
      x = NULL
    ) +
    bar_theme +
    guides(
      fill = guide_legend(
        nrow = 1,
        byrow = TRUE
      )
    )

  if (compact_style) {

    compact_top_margin <- if (panel_index == 1) {
      1
    } else {
      METRIC_PLOT_GAP_PT / 2
    }

    compact_bottom_margin <- if (panel_index == panel_count) {
      BOTTOM_TO_LEGEND_GAP_PT
    } else {
      METRIC_PLOT_GAP_PT / 2
    }

    p <- p +
      theme(
        axis.text.x = element_text(
          size = COMPACT_DATASET_X_TEXT_SIZE,
          angle = 55,
          hjust = 1,
          vjust = 1,
          margin = margin(t = 1),
          lineheight = 0.85
        ),

        axis.text.y = element_text(
          size = COMPACT_AXIS_TEXT_SIZE
        ),

        axis.title.y = element_text(
          size = COMPACT_AXIS_TITLE_SIZE
        ),

        strip.text.x = element_text(
          size = COMPACT_STRIP_TEXT_SIZE,
          face = "plain"
        ),

        panel.spacing.x = unit(
          0.18,
          "cm"
        ),

        legend.text = element_text(
          size = COMPACT_LEGEND_TEXT_SIZE
        ),

        legend.key.height = unit(
          0.26,
          "cm"
        ),

        legend.key.width = unit(
          0.42,
          "cm"
        ),

        plot.margin = margin(
          compact_top_margin,
          2,
          compact_bottom_margin,
          2
        )
      )
  }

  if (!show_x_labels) {
    # Remove the unused axes completely so the shared-axis panels close up.
    p <- p +
      theme(
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()
      )
  }

  if (!show_tissue_labels) {
    p <- p +
      theme(
        strip.text.x = element_blank()
      )
  }

  p
}

save_pdf <- function(
  plot_obj,
  out_file,
  width,
  height,
  crop = TRUE
) {

  if (!crop) {

    ggsave(
      out_file,
      plot_obj,
      width = width,
      height = height,
      device = cairo_pdf
    )

    message(
      "Saved: ",
      out_file
    )

    return(
      invisible(NULL)
    )
  }

  tmp_file <- sub(
    "\\.pdf$",
    ".tmp.pdf",
    out_file
  )

  ggsave(
    tmp_file,
    plot_obj,
    width = width,
    height = height,
    device = cairo_pdf
  )

  crop_output <- system2(
    "pdfcrop",
    c(
      tmp_file,
      out_file
    ),
    stdout = TRUE,
    stderr = TRUE
  )

  crop_status <- attr(
    crop_output,
    "status"
  )

  if (is.null(crop_status)) {
    crop_status <- 0
  }

  if (crop_status != 0) {

    warning(
      "pdfcrop failed; saving the uncropped PDF instead.\n",
      paste(
        crop_output,
        collapse = "\n"
      )
    )

    invisible(
      file.copy(
        tmp_file,
        out_file,
        overwrite = TRUE
      )
    )
  }

  invisible(
    file.remove(
      tmp_file
    )
  )

  message(
    "Saved: ",
    out_file
  )
}

render_stacked_figure <- function(
  out_file,
  separate_x_ticks = FALSE,
  width,
  height,
  crop = TRUE
) {

  show_x_labels <- if (separate_x_ticks) {
    rep(
      TRUE,
      length(METRIC_LEVELS)
    )
  } else {
    seq_along(METRIC_LEVELS) ==
      length(METRIC_LEVELS)
  }

  # Dataset order and tissue boundaries are identical across metrics, so one
  # set of tissue headings above the F1 panel labels the entire stack.
  show_tissue_labels <-
    seq_along(METRIC_LEVELS) == 1

  panels <- Map(
    function(
      metric_name,
      show_labels,
      show_tissues,
      index
    ) {

      make_dataset_panel(
        metric_name,
        show_x_labels = show_labels,
        show_tissue_labels = show_tissues,
        panel_index = index,
        panel_count = length(METRIC_LEVELS),
        compact_style = TRUE
      )
    },
    METRIC_LEVELS,
    show_x_labels,
    show_tissue_labels,
    seq_along(METRIC_LEVELS)
  )

  combined <- (
    wrap_plots(
      panels,
      ncol = 1
    ) +
      plot_layout(
        heights = rep(
          1,
          length(panels)
        ),
        guides = "collect"
      )
  ) &
    theme(
      legend.position = "bottom",
      legend.box.spacing = unit(
        0,
        "pt"
      ),
      legend.margin = margin(
        t = 0,
        r = 0,
        b = 0,
        l = 0
      )
    )

  save_pdf(
    combined,
    out_file,
    width = width,
    height = height,
    crop = crop
  )
}

# =========================================================
# EXPORT FIGURES
# =========================================================

render_stacked_figure(
  file.path(
    out_dir,
    "zeroshot_metrics_3row_shared_xticks.pdf"
  ),
  separate_x_ticks = FALSE,
  width = SHARED_FIG_WIDTH_IN,
  height = SHARED_FIG_HEIGHT_IN,
  crop = TRUE
)

render_stacked_figure(
  file.path(
    out_dir,
    "zeroshot_metrics_3row_separate_xticks.pdf"
  ),
  separate_x_ticks = TRUE,
  width = SEPARATE_FIG_WIDTH_IN,
  height = SEPARATE_FIG_HEIGHT_IN
)