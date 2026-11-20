#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(cowplot)
  library(grid)
})

args <- commandArgs(trailingOnly = FALSE)
script_arg <- args[grep("--file=", args)]

if (length(script_arg) > 0) {
  script_path <- normalizePath(sub("--file=", "", script_arg))
}

project_root <- "./"
data_root <- file.path(project_root, "outputs", "tables")
out_dir <- file.path("figure")

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# =========================================================
# FONT / SPACING CONTROLS
# =========================================================

AXIS_TEXT_SIZE       <- 6.5
Y_TITLE_SIZE         <- 8
Y_TITLE_RIGHT_GAP    <- 0
PLOT_LEFT_MARGIN     <- 7

LEGEND_TEXT_SIZE     <- 9
LEGEND_KEY_WIDTH     <- 1.2

ROW_LABEL_SIZE       <- 2.65

POINT_SIZE           <- 1.8
LINE_WIDTH           <- 0.6

# =========================================================
# LOAD DATA
# =========================================================

df <- read_csv(
  file.path(data_root, "Fig3_topK_celltype_metrics.csv"),
  show_col_types = FALSE
)

df <- read_csv(
  file.path(data_root, "Fig3_topK_celltype_metrics.csv"),
  show_col_types = FALSE
)

df$config <- recode(
  df$config,
  "scLEAD" = "scLEAP"
)

df$config <- factor(
  df$config,
  levels = c("scLEAP", "scTab", "scFoundation", "CellFM", "SCANPY", "Seurat")
)
method_colors <- c(
  "scLEAP"        = "#E64B35FF",
  "scTab"         = "#4DBBD5FF",
  "scFoundation" = "#3C5488FF",
  "CellFM"        = "#00A087FF",
  "SCANPY"        = "#8491B4FF",
  "Seurat"        = "#91D1C2FF"
)

method_shapes <- c(
  "scLEAP"        = 16,
  "scTab"         = 17,
  "scFoundation" = 15,
  "CellFM"        = 18,
  "SCANPY"        = 4,
  "Seurat"        = 8
)

# =========================================================
# BASE THEME
# =========================================================

base_theme <- theme_minimal(base_size = 9) +
  theme(
    plot.title = element_blank(),

    axis.text = element_text(
      size = AXIS_TEXT_SIZE,
      color = "grey30"
    ),

    axis.title.x = element_blank(),

    axis.line.x = element_line(color = "grey40", linewidth = 0.4),
    axis.line.y = element_line(color = "grey40", linewidth = 0.4),
    axis.ticks  = element_line(color = "grey40", linewidth = 0.3),

    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),

    legend.position = "none",

    plot.margin = margin(
      t = 2,
      r = 3,
      b = 2,
      l = PLOT_LEFT_MARGIN
    )
  )

# =========================================================
# PANEL FUNCTION
# =========================================================
make_panel <- function(data, metric_col, title) {

  p <- ggplot(
    data,
    aes(
      x = topK,
      y = .data[[metric_col]],
      color = config,
      shape = config,
      group = config
    )
  ) +
    geom_line(linewidth = LINE_WIDTH) +
    geom_point(size = POINT_SIZE) +

    scale_color_manual(values = method_colors, drop = FALSE) +
    scale_shape_manual(values = method_shapes, drop = FALSE) +

    scale_x_continuous(
      breaks = seq(10, 70, 10)
    ) +
    coord_cartesian(
      xlim = c(10, 70)
    ) +

    labs(y = title) +

    base_theme +

    theme(
      axis.title.y = element_text(
        size = Y_TITLE_SIZE,
        margin = margin(r = Y_TITLE_RIGHT_GAP)
      )
    )

  return(p)
}

# =========================================================
# BUILD PANELS
# =========================================================

p1 <- make_panel(df, "cent_acc_macro", "Centroid-based Agreement")

p2 <- make_panel(df, "sil", "Silhouette Score")

p3 <- make_panel(df, "dbi_pm", "Davies–Bouldin Index")

# =========================================================
# LEGEND
# =========================================================

p_legend <- ggplot(
  df,
  aes(
    x = topK,
    y = cent_acc_macro,
    color = config,
    shape = config,
    group = config
  )
) +
  geom_line(linewidth = LINE_WIDTH) +
  geom_point(size = POINT_SIZE) +

  scale_color_manual(
    values = method_colors,
    drop = FALSE,
    name = NULL
  ) +
  scale_shape_manual(
    values = method_shapes,
    drop = FALSE,
    name = NULL
  ) +

  guides(
    color = guide_legend(
      nrow = 1,
      byrow = TRUE
    ),
    shape = guide_legend(
      nrow = 1,
      byrow = TRUE
    )
  ) +

  theme_void() +
  theme(
    legend.position = "top",

    legend.text = element_text(
      size = LEGEND_TEXT_SIZE
    ),

    legend.key.width = unit(
      LEGEND_KEY_WIDTH,
      "cm"
    ),

    legend.background = element_blank(),
    legend.key = element_rect(fill = NA, colour = NA),
    legend.margin = margin(t = 0, r = 0, b = 0, l = 0),
    legend.box.margin = margin(t = -4, r = 0, b = -4, l = 0)
  )

legend_plot_grob <- ggplotGrob(p_legend)
legend_index <- which(legend_plot_grob$layout$name == "guide-box-top")

if (length(legend_index) != 1) {
  stop("Could not locate the top legend in the ggplot gtable")
}

legend_grob <- legend_plot_grob$grobs[[legend_index]]

# =========================================================
# ROW LABEL
# =========================================================

bottom_label <- cowplot::ggdraw() +
  cowplot::draw_label(
    "Most Abundant Cell Types",
    size = ROW_LABEL_SIZE * 3.2,
    x = 0.5,
    y = 0.5,
    hjust = 0.5,
    vjust = 0.5
  ) +
  theme(
    plot.margin = margin(
      t = -4,
      r = 0,
      b = 2,
      l = 0
    )
  )
# =========================================================
# ALIGN PANELS
# =========================================================

plots <- list(p1, p2, p3)

# Convert plots to gtables
grobs <- lapply(plots, ggplotGrob)

# Force every plot to use the same column widths
# This aligns y-axis titles, y-axis tick labels, panels, and right margins.
max_widths <- do.call(
  grid::unit.pmax,
  lapply(grobs, function(g) g$widths)
)

grobs <- lapply(
  grobs,
  function(g) {
    g$widths <- max_widths
    g
  }
)

# Force every plot to use the same row heights
# This aligns x-axis labels and panel heights.
max_heights <- do.call(
  grid::unit.pmax,
  lapply(grobs, function(g) g$heights)
)

grobs <- lapply(
  grobs,
  function(g) {
    g$heights <- max_heights
    g
  }
)

panel_row <- cowplot::plot_grid(
  plotlist = grobs,
  nrow = 1,
  align = "none",
  rel_widths = c(1, 1, 1)
)

# =========================================================
# COMBINE PANELS
# =========================================================

combined <- cowplot::plot_grid(
  legend_grob,
  panel_row,
  bottom_label,
  ncol = 1,
  rel_heights = c(0.25, 1, 0.12)
)

# =========================================================
# SAVE
# =========================================================

tmp_file <- file.path(out_dir, "fig3_embedding_metrics.tmp.pdf")
out_file <- file.path(out_dir, "fig3_embedding_metrics.pdf")

ggsave(
  tmp_file,
  combined,
  width = 7.5,
  # Preserve the original per-panel proportions after changing from two rows
  # to one while giving the legend and shared x-axis label enough room.
  height = 2.074,
  device = cairo_pdf
)

pdfcrop_bin <- "/usr/bin/pdfcrop"

if (!file.exists(pdfcrop_bin)) {
  stop("pdfcrop was not found at /usr/bin/pdfcrop")
}

crop_output <- system2(
  pdfcrop_bin,
  args = c(
    "--margins",
    shQuote("0 0 0 0"),
    tmp_file,
    out_file
  ),
  stdout = TRUE,
  stderr = TRUE
)

crop_status <- attr(crop_output, "status")

if (is.null(crop_status)) {
  crop_status <- 0
}

if (crop_status != 0) {
  message("pdfcrop failed. Full output:")
  message(paste(crop_output, collapse = "\n"))

  message("Saving uncropped PDF instead.")
  invisible(file.copy(tmp_file, out_file, overwrite = TRUE))
}

invisible(file.remove(tmp_file))

message("Saved: ", out_file)
