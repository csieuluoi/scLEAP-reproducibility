
#!/usr/bin/env Rscript

# ============================================================
# Top 10 most abundant cell types
# Separate plots for:
#   1. Macro F1
#   2. Recall
#   3. Precision
#
# NOTE:
#   Internal method name remains "scLEAD"
#   Figure legend displays "scLEAP"
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
  library(grid)
})

# ============================================================
# PATHS
# ============================================================

project_root <- "./"

boot_dir <- file.path(
  project_root,
  "outputs", "tables"
)

out_dir <- "figure"

dir.create(
  out_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

bootstrap_fine_csv <- file.path(
  boot_dir,
  "zeroshot_per_celltype_bootstrap.csv"
)

# ============================================================
# SETTINGS
# ============================================================

N_TOP_CELLTYPES <- 10

FIG_WIDTH  <- 5.4
FIG_HEIGHT <- 4.2

BASE_FONT_SIZE   <- 12
AXIS_TEXT_SIZE   <- 10.5
AXIS_TITLE_SIZE  <- 12
LEGEND_TEXT_SIZE <- 11

BAR_WIDTH          <- 0.60
DODGE_WIDTH        <- 0.69
ERRORBAR_WIDTH     <- 0.18
ERRORBAR_LINEWIDTH <- 0.30

# ============================================================
# METHOD COLORS / ORDER
# ============================================================

# Keep original method names here because these are the
# names stored in the input CSV.
method_colors <- c(
  "scLEAD"        = "#E64B35FF",
  "LangCell"      = "#4DBBD5FF",
  "CellWhisperer" = "#00A087FF"
)

method_levels <- names(method_colors)

# Display names used only in the figure legend
method_labels <- c(
  "scLEAD"        = "scLEAP",
  "LangCell"      = "LangCell",
  "CellWhisperer" = "CellWhisperer"
)

# ============================================================
# LOAD DATA
# ============================================================

boot_fine <- read_csv(
  bootstrap_fine_csv,
  show_col_types = FALSE
)

# ============================================================
# FIND TOP 10 MOST ABUNDANT CELL TYPES
#
# n_cells is the same across methods, so use scLEAD only
# to avoid counting each cell type multiple times.
# ============================================================

ct_support <- boot_fine %>%
  filter(method == "scLEAD") %>%
  arrange(desc(n_cells)) %>%
  pull(cell_type)

top_cts <- head(
  ct_support,
  N_TOP_CELLTYPES
)

cat("\nTop 10 most abundant cell types:\n")
print(top_cts)

# ============================================================
# RESHAPE METRICS
# ============================================================

df_fine <- boot_fine %>%
  pivot_longer(
    cols = matches(
      "^(F1|Precision|Recall)_(observed|ci_lo|ci_hi)$"
    ),
    names_to = c("metric", ".value"),
    names_pattern =
      "^(F1|Precision|Recall)_(observed|ci_lo|ci_hi)$"
  ) %>%
  rename(
    value = observed,
    lo    = ci_lo,
    hi    = ci_hi
  ) %>%
  mutate(
    metric = factor(
      metric,
      levels = c(
        "F1",
        "Recall",
        "Precision"
      )
    ),
    method = factor(
      method,
      levels = method_levels
    ),
    ymin = pmax(
      0,
      lo
    ),
    ymax = pmin(
      1,
      hi
    )
  ) %>%
  filter(
    cell_type %in% top_cts
  ) %>%
  mutate(
    # Most abundant cell type appears first / on the left
    cell_type = factor(
      cell_type,
      levels = top_cts
    )
  )

# ============================================================
# PRETTY CELL-TYPE LABELS
# ============================================================

pretty_ct <- function(x) {

  x <- as.character(x)

  x <- gsub(
    "type b pancreatic cell",
    "beta cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "pancreatic a cell",
    "alpha cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "pancreatic d cell",
    "delta cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "pancreatic pp cell",
    "PP cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "pancreatic acinar cell",
    "acinar cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "pancreatic ductal cell",
    "ductal cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "kidney proximal tubule epithelial cell",
    "proximal tubule cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "thick ascending limb of loop of henle epithelial cell",
    "thick asc. limb cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "distal convoluted tubule epithelial cell",
    "distal conv. tubule cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "mhc class ii-positive cell",
    "MHC II+ cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "pulmonary alveolar type 2 cell",
    "alveolar type 2 cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "renal alpha-intercalated cell",
    "renal alpha-intercalated cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "renal beta-intercalated cell",
    "renal beta-intercalated cell",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "kidney leukocyte cell",
    "kidney leukocyte",
    x,
    fixed = TRUE
  )

  x <- gsub(
    "^t cell$",
    "T cell",
    x,
    ignore.case = TRUE
  )

  tools::toTitleCase(x)
}

# ============================================================
# PLOT FUNCTION
# ============================================================

make_top10_plot <- function(
  metric_name,
  y_label,
  out_file
) {

  df_plot <- df_fine %>%
    filter(
      metric == metric_name
    )

  p <- ggplot(
    df_plot,
    aes(
      x = cell_type,
      y = value,
      fill = method
    )
  ) +

    # --------------------------------------------------------
    # Bars
    # --------------------------------------------------------
    geom_col(
      position = position_dodge(
        width = DODGE_WIDTH
      ),
      width = BAR_WIDTH
    ) +

    # --------------------------------------------------------
    # Bootstrap confidence intervals
    # --------------------------------------------------------
    geom_errorbar(
      aes(
        ymin = ymin,
        ymax = ymax,
        group = method
      ),
      position = position_dodge(
        width = DODGE_WIDTH
      ),
      width = ERRORBAR_WIDTH,
      linewidth = ERRORBAR_LINEWIDTH,
      colour = "grey30"
    ) +

    # --------------------------------------------------------
    # Colors + displayed labels
    # --------------------------------------------------------
    scale_fill_manual(
      values = method_colors,
      breaks = method_levels,
      labels = method_labels,
      drop = FALSE,
      name = NULL
    ) +

    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------
    scale_y_continuous(
      limits = c(
        0,
        1
      ),
      breaks = seq(
        0,
        1,
        0.2
      ),
      expand = c(
        0,
        0
      )
    ) +

    # --------------------------------------------------------
    # Cell-type labels
    # --------------------------------------------------------
    scale_x_discrete(
      labels = pretty_ct
    ) +

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    labs(
      title = sprintf(
        "Top %d Most Abundant Cell Types",
        N_TOP_CELLTYPES
      ),
      x = NULL,
      y = y_label
    ) +

    # --------------------------------------------------------
    # Theme
    # --------------------------------------------------------
    theme_minimal(
      base_size = BASE_FONT_SIZE
    ) +

    theme(

      # Axis lines
      axis.line = element_line(
        colour = "grey50",
        linewidth = 0.35
      ),

      # Grid
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),

      # Title
      plot.title = element_text(
        size = BASE_FONT_SIZE,
        face = "bold",
        hjust = 0.5
      ),

      # X-axis labels
      #
      # Negative bottom margin reduces the space between
      # the rotated x-axis labels and the legend.
      axis.text.x = element_text(
        size = AXIS_TEXT_SIZE,
        angle = 45,
        hjust = 0.95,
        vjust = 1,
        margin = margin(
          t = 1,
          b = -3
        ),
        lineheight = 0.9
      ),

      # Y-axis labels
      axis.text.y = element_text(
        size = AXIS_TEXT_SIZE
      ),

      axis.title.y = element_text(
        size = AXIS_TITLE_SIZE
      ),

      # ------------------------------------------------------
      # Legend
      # ------------------------------------------------------
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.title = element_blank(),

      legend.text = element_text(
        size = LEGEND_TEXT_SIZE
      ),

      # Pull legend upward toward x-axis labels
      legend.margin = margin(
        t = -8,
        r = 0,
        b = 0,
        l = 0
      ),

      # Remove extra outer legend-box spacing
      legend.box.margin = margin(
        t = -4,
        r = 0,
        b = 0,
        l = 0
      ),

      # Remove vertical spacing within legend area
      legend.spacing.y = unit(
        0,
        "pt"
      ),

      # Make legend keys slightly more compact
      legend.key.height = unit(
        0.45,
        "cm"
      ),

      # ------------------------------------------------------
      # Overall plot margin
      # ------------------------------------------------------
      plot.margin = margin(
        t = 6,
        r = 6,
        b = 2,
        l = 6
      )
    ) +

    guides(
      fill = guide_legend(
        nrow = 1,
        byrow = TRUE
      )
    )

  # ==========================================================
  # SAVE
  # ==========================================================

  ggsave(
    filename = out_file,
    plot = p,
    width = FIG_WIDTH,
    height = FIG_HEIGHT,
    device = "pdf"
  )

  message(
    "Saved: ",
    out_file
  )

  return(p)
}

# ============================================================
# CREATE THREE SEPARATE FIGURES
# ============================================================

# 1. Macro F1
p_macro_f1 <- make_top10_plot(
  metric_name = "F1",
  y_label = "Macro F1",
  out_file = file.path(
    out_dir,
    "supp_zeroshot_most_abbundant_ct_F1.pdf"
  )
)

# 2. Recall
p_recall <- make_top10_plot(
  metric_name = "Recall",
  y_label = "Recall",
  out_file = file.path(
    out_dir,
    "supp_zeroshot_most_abbundant_ct_Recall.pdf"
  )
)

# 3. Precision
p_precision <- make_top10_plot(
  metric_name = "Precision",
  y_label = "Precision",
  out_file = file.path(
    out_dir,
    "supp_zeroshot_most_abbundant_ct_Precision.pdf"
  )
)

message("\nDone.")
