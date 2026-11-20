#!/usr/bin/env Rscript
# Over/under-prediction plot — every cell type that either appears in truth
# or in any method's predictions is shown as a labelled dot. No cell-type
# filtering. Axes log-scale with +1 to handle zero-predicted classes.
#
# Output: figure/zeroshot_pred_count_bias.pdf

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
  library(ggrepel)
  library(scales)
  library(patchwork)
})

args <- commandArgs(trailingOnly = FALSE)

script_path <- normalizePath(
  sub("--file=", "", args[grep("--file=", args)])
)

project_root <- "./"

preds_dir <- file.path(
  project_root,
  "outputs", "tables",
  "method_predictions"
)

out_dir <- file.path("figure")

# =========================
# FONT / SIZE CONTROLS
# =========================

BASE_FONT_SIZE <- 11 + 1.5

METHOD_TITLE_SIZE <- 12 + 0.5
METHOD_TITLE_FACE <- "bold"
METHOD_TITLE_MARGIN_B <- -2

AXIS_TITLE_SIZE <- 12 + 1
AXIS_TICK_SIZE  <- 10 + 1

CELLTYPE_LABEL_SIZE <- 4
R2_TEXT_SIZE        <- 4.5

POINT_SIZE <- 2.15

GRID_LINE_WIDTH <- 0.25
AXIS_LINE_WIDTH <- 0.35
TICK_LENGTH_PT  <- 2.5

SAVE_WIDTH  <- 0.4 * 12
SAVE_HEIGHT <- 0.4 * 23.4

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

label_km <- function(x) {
  vapply(
    x,
    function(v) {
      if (is.na(v)) return(NA_character_)
      if (v < 1000) {
        return(format(v, big.mark = ",", trim = TRUE))
      }
      if (v < 1e6) {
        return(
          paste0(
            format(
              round(v / 1e3),
              big.mark = ",",
              trim = TRUE
            ),
            "K"
          )
        )
      }

      paste0(
        format(
          round(v / 1e6),
          big.mark = ",",
          trim = TRUE
        ),
        "M"
      )
    },
    character(1)
  )
}

group_order <- c(
  "Pancreatic endocrine",
  "Pancreatic exocrine & stromal",
  "Renal tubule",
  "Renal parietal & urothelial",
  "Liver",
  "Airway & alveolar epithelial",
  "T / NK / innate lymphoid",
  "B / plasma",
  "Myeloid",
  "Stromal, endothelial & blood",
  "Other"
)

group_colors <- setNames(
  c(
    "#E41A1C",
    "#FB8072",
    "#377EB8",
    "#80B1D3",
    "#984EA3",
    "#4DAF4A",
    "#FF7F00",
    "#FFD92F",
    "#A65628",
    "#666666",
    "#BDBDBD"
  ),
  group_order
)

group_map <- c(
  "pancreatic a cell" = "Pancreatic endocrine",
  "type b pancreatic cell" = "Pancreatic endocrine",
  "pancreatic d cell" = "Pancreatic endocrine",
  "pancreatic pp cell" = "Pancreatic endocrine",
  "pancreatic epsilon cell" = "Pancreatic endocrine",

  "pancreatic acinar cell" = "Pancreatic exocrine & stromal",
  "pancreatic ductal cell" = "Pancreatic exocrine & stromal",
  "pancreatic stellate cell" = "Pancreatic exocrine & stromal",

  "kidney proximal tubule epithelial cell" = "Renal tubule",
  "distal convoluted tubule epithelial cell" = "Renal tubule",
  "thick ascending limb of loop of henle epithelial cell" = "Renal tubule",
  "renal principal cell" = "Renal tubule",
  "renal alpha-intercalated cell" = "Renal tubule",
  "renal beta-intercalated cell" = "Renal tubule",

  "podocyte" = "Renal parietal & urothelial",
  "parietal epithelial cell" = "Renal parietal & urothelial",
  "urothelial cell" = "Renal parietal & urothelial",

  "hepatocyte" = "Liver",
  "cholangiocyte" = "Liver",
  "hepatic stellate cell" = "Liver",
  "endothelial cell of pericentral hepatic sinusoid" = "Liver",
  "endothelial cell of periportal hepatic sinusoid" = "Liver",
  "kupffer cell" = "Liver",

  "basal cell" = "Airway & alveolar epithelial",
  "ciliated cell" = "Airway & alveolar epithelial",
  "club cell" = "Airway & alveolar epithelial",
  "goblet cell" = "Airway & alveolar epithelial",
  "pulmonary alveolar type 2 cell" = "Airway & alveolar epithelial",
  "pulmonary ionocyte" = "Airway & alveolar epithelial",
  "lung neuroendocrine cell" = "Airway & alveolar epithelial",
  "tuft cell" = "Airway & alveolar epithelial",
  "secretory cell" = "Airway & alveolar epithelial",
  "mucus secreting cell" = "Airway & alveolar epithelial",
  "epithelial cell" = "Airway & alveolar epithelial",
  "airway submucosal gland collecting duct epithelial cell" =
    "Airway & alveolar epithelial",

  "t cell" = "T / NK / innate lymphoid",
  "alpha-beta t cell" = "T / NK / innate lymphoid",
  "gamma-delta t cell" = "T / NK / innate lymphoid",
  "cycling t cell" = "T / NK / innate lymphoid",
  "natural killer cell" = "T / NK / innate lymphoid",

  "b cell" = "B / plasma",
  "mature b cell" = "B / plasma",
  "plasma cell" = "B / plasma",

  "monocyte" = "Myeloid",
  "macrophage" = "Myeloid",
  "cycling macrophage" = "Myeloid",
  "inflammatory macrophage" = "Myeloid",
  "neutrophil" = "Myeloid",
  "conventional dendritic cell" = "Myeloid",
  "plasmacytoid dendritic cell" = "Myeloid",
  "mast cell" = "Myeloid",
  "mhc class ii-positive cell" = "Myeloid",
  "kidney leukocyte cell" = "Myeloid",

  "fibroblast" = "Stromal, endothelial & blood",
  "smooth muscle cell" = "Stromal, endothelial & blood",
  "endothelial cell" = "Stromal, endothelial & blood",
  "lymphatic endothelial cell" = "Stromal, endothelial & blood",
  "platelet" = "Stromal, endothelial & blood",
  "erythroid lineage cell" = "Stromal, endothelial & blood",
  "mesothelial cell" = "Stromal, endothelial & blood",
  "myofibroblast" = "Stromal, endothelial & blood",
  "schwann cell" = "Stromal, endothelial & blood"
)

pretty_ct_aliases <- c(
  "pancreatic a cell" = "alpha",
  "type b pancreatic cell" = "beta",
  "pancreatic d cell" = "delta",
  "pancreatic pp cell" = "PP",
  "pancreatic epsilon cell" = "epsilon",
  "pancreatic acinar cell" = "acinar",
  "pancreatic ductal cell" = "ductal",
  "pancreatic stellate cell" = "panc. stellate",
  "kidney proximal tubule epithelial cell" = "prox. tubule",
  "thick ascending limb of loop of henle epithelial cell" = "thick asc. limb",
  "distal convoluted tubule epithelial cell" = "distal tubule",
  "renal principal cell" = "renal principal",
  "renal alpha-intercalated cell" = "alpha-interc.",
  "renal beta-intercalated cell" = "beta-interc.",
  "parietal epithelial cell" = "parietal epith.",
  "urothelial cell" = "urothelial",
  "airway submucosal gland collecting duct epithelial cell" = "airway gland duct",
  "pulmonary alveolar type 2 cell" = "alveolar T2",
  "lung neuroendocrine cell" = "lung neuro.",
  "mhc class ii-positive cell" = "MHC II+",
  "conventional dendritic cell" = "cDC",
  "plasmacytoid dendritic cell" = "pDC",
  "erythroid lineage cell" = "erythroid",
  "natural killer cell" = "NK",
  "kidney leukocyte cell" = "kidney leuk.",
  "endothelial cell of pericentral hepatic sinusoid" = "pericentral hep. EC",
  "endothelial cell of periportal hepatic sinusoid" = "periportal hep. EC",
  "lymphatic endothelial cell" = "lymphatic EC",
  "inflammatory macrophage" = "inflammatory macro.",
  "cycling macrophage" = "cycling macro.",
  "cycling t cell" = "cycling T",
  "alpha-beta t cell" = "alpha-beta T",
  "gamma-delta t cell" = "gamma-delta T",
  "mature b cell" = "mature B",
  "mucus secreting cell" = "mucus secr."
)

pretty_ct <- function(x) {
  ifelse(
    x %in% names(pretty_ct_aliases),
    pretty_ct_aliases[x],
    x
  )
}

# =========================
# LOAD DATA
# =========================

preds <- bind_rows(
  lapply(
    list.files(
      preds_dir,
      pattern = "\\.csv$",
      full.names = TRUE
    ),
    function(f) {
      d <- read_csv(
        f,
        show_col_types = FALSE
      )

      # Keep filename-derived method names unchanged.
      d$method <- sub(
        "\\.csv$",
        "",
        basename(f)
      )

      d
    }
  )
) %>%
  mutate(
    method = factor(
      method,
      levels = method_levels
    )
  )

counts <- preds %>%
  count(
    method,
    cell_type,
    name = "n_true"
  ) %>%
  full_join(
    preds %>%
      count(
        method,
        predicted_cell_type,
        name = "n_pred"
      ) %>%
      rename(
        cell_type = predicted_cell_type
      ),
    by = c(
      "method",
      "cell_type"
    )
  ) %>%
  mutate(
    n_true = replace_na(
      n_true,
      0
    ),
    n_pred = replace_na(
      n_pred,
      0
    ),
    group = ifelse(
      cell_type %in% names(group_map),
      group_map[cell_type],
      "Other"
    ),
    group = factor(
      group,
      levels = group_order
    ),
    label = pretty_ct(
      cell_type
    )
  )

r2_df <- counts %>%
  mutate(
    log_true = log10(n_true + 1),
    log_pred = log10(n_pred + 1)
  ) %>%
  group_by(method) %>%
  summarise(
    r2 = summary(
      lm(log_pred ~ log_true)
    )$r.squared,
    .groups = "drop"
  )

cat(
  sprintf(
    "Total distinct cell types across truth ∪ prediction: %d\n",
    n_distinct(counts$cell_type)
  )
)

cat("Methods:\n")
print(table(counts$method))

cat("Regression R2 on log10(count + 1) scale:\n")
print(r2_df)

n_labels_per_facet <- 10

label_df <- counts %>%
  mutate(
    off_diag = abs(
      log10(
        (n_pred + 1) /
          (n_true + 1)
      )
    )
  ) %>%
  group_by(method) %>%
  slice_max(
    off_diag,
    n = n_labels_per_facet,
    with_ties = FALSE
  ) %>%
  ungroup()

axis_lo <- 1
axis_hi <- max(
  counts$n_true,
  counts$n_pred
) + 1

axis_breaks <- c(
  1,
  10,
  100,
  1000,
  10000,
  1e5
)

base_theme <- theme_minimal(
  base_size = BASE_FONT_SIZE
) +
  theme(
    panel.grid.minor = element_blank(),

    panel.grid.major = element_line(
      colour = "grey94",
      linewidth = GRID_LINE_WIDTH
    ),

    axis.line = element_line(
      colour = "grey50",
      linewidth = AXIS_LINE_WIDTH
    ),

    axis.ticks = element_line(
      colour = "grey50",
      linewidth = AXIS_LINE_WIDTH
    ),

    axis.ticks.length = unit(
      TICK_LENGTH_PT,
      "pt"
    ),

    axis.title.y = element_text(
      size = AXIS_TITLE_SIZE,
      margin = margin(r = -5)
    ),

    plot.title = element_text(
      size = METHOD_TITLE_SIZE,
      face = METHOD_TITLE_FACE,
      hjust = 0.5,
      margin = margin(
        b = METHOD_TITLE_MARGIN_B
      )
    ),

    axis.title = element_text(
      size = AXIS_TITLE_SIZE
    ),

    axis.text = element_text(
      size = AXIS_TICK_SIZE
    ),

    legend.position = "none",

    plot.margin = margin(
      3,
      5,
      3,
      5
    )
  )

make_facet <- function(
  method_name,
  show_y_labels,
  show_y_title,
  show_x_title
) {

  sub <- counts %>%
    filter(
      method == method_name
    )

  lab <- label_df %>%
    filter(
      method == method_name
    )

  r2_val <- r2_df %>%
    filter(
      method == method_name
    ) %>%
    pull(r2)

  # Convert internal name to display name only for plot title.
  display_method_name <- unname(
    method_display_labels[method_name]
  )

  p <- ggplot(
    sub,
    aes(
      x = n_true + 1,
      y = n_pred + 1,
      colour = group
    )
  ) +
    geom_abline(
      linetype = "dashed",
      colour = "grey55"
    ) +
    geom_point(
      size = POINT_SIZE,
      alpha = 0.85
    ) +
    annotate(
      "text",
      x = axis_lo * 1.4,
      y = axis_hi / 1.6,
      label = sprintf(
        "R² = %.2f",
        r2_val
      ),
      hjust = 0,
      size = R2_TEXT_SIZE
    ) +
    ggrepel::geom_text_repel(
      data = lab,
      aes(
        label = label
      ),
      size = CELLTYPE_LABEL_SIZE,
      max.overlaps = 40,
      min.segment.length = 0.1,
      segment.size = 0.15,
      segment.colour = "grey70",
      show.legend = FALSE
    ) +
    scale_x_log10(
      labels = label_km,
      breaks = axis_breaks,
      limits = c(
        axis_lo,
        axis_hi
      )
    ) +
    scale_y_log10(
      labels = label_km,
      breaks = axis_breaks,
      limits = c(
        axis_lo,
        axis_hi
      )
    ) +
    scale_colour_manual(
      values = group_colors,
      drop = FALSE
    ) +
    labs(
      title = display_method_name,
      x = if (show_x_title) {
        "True cell number"
      } else {
        NULL
      },
      y = if (show_y_title) {
        "Predicted cell number"
      } else {
        NULL
      }
    ) +
    base_theme

  if (!show_y_labels) {
    p <- p +
      theme(
        axis.text.y = element_blank()
      )
  }

  p
}

methods_here <- levels(
  counts$method
)

middle_idx <- ceiling(
  length(methods_here) / 2
)

plots <- lapply(
  seq_along(methods_here),
  function(i) {
    make_facet(
      methods_here[i],
      show_y_labels = TRUE,
      show_y_title = (i == middle_idx),
      show_x_title = (i == length(methods_here))
    )
  }
)

p <- Reduce(`/`, plots) +
  plot_layout(
    heights = rep(
      1,
      length(plots)
    )
  )

tmp_file <- file.path(
  out_dir,
  "zeroshot_pred_count_bias.tmp.pdf"
)

out_file <- file.path(
  out_dir,
  "zeroshot_pred_count_bias.pdf"
)

ggsave(
  tmp_file,
  p,
  width = SAVE_WIDTH,
  height = SAVE_HEIGHT,
  device = "pdf"
)

system2(
  "pdfcrop",
  c(
    tmp_file,
    out_file
  ),
  stdout = FALSE,
  stderr = FALSE
)

file.remove(
  tmp_file
)

message(
  "Saved: ",
  out_file
)