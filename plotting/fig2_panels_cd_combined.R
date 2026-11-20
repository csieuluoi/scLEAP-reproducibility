#!/usr/bin/env Rscript
# Figure 2 Panels C+D: Render together for alignment, then crop into separate PDFs
# Outputs: figure/zeroshot_pancreas_results.pdf, figure/zeroshot_blood_results.pdf

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(patchwork)
  library(cowplot)
  library(ggtext)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- normalizePath(sub("--file=", "", args[grep("--file=", args)]))
project_root <- "./"
fm_root <- file.path("./resources/scLEAP/results/foundation_model_results")
out_dir <- "./figure"

# NPG colors
npg_colors <- c(
  "#E64B35FF", "#4DBBD5FF", "#00A087FF", "#3C5488FF", "#F39B7FFF",
  "#8491B4FF", "#91D1C2FF", "#DC0000FF", "#7E6148FF", "#B09C85FF",
  "#5f7de7"
)

# Cell-type-specific color pins for the zero-shot failure-mode story.
# Principal (the ground-truth majority that scLEAD over-predicts as urothelial)
# and Urothelial (the over-predicted attractor) get the two most contrasting
# NPG hues so the red/blue swap between ground-truth and predicted UMAPs is
# visible at a glance.
pinned_cell_colors <- c(
  "Principal"           = "#E64B35FF",  # vivid red
  "Urothelial"          = "#3C5488FF",  # dark blue
  "Thick Ascending Limb" = "#FF7F00FF"  # bright orange (complement of dark blue)
)

clean_ct <- function(x) {
  # Pancreas
  x <- gsub("pancreatic a cell", "Alpha", x)
  x <- gsub("pancreatic acinar cell", "Acinar", x)
  x <- gsub("pancreatic d cell", "Delta", x)
  x <- gsub("pancreatic ductal cell", "Ductal", x)
  x <- gsub("pancreatic epsilon cell", "Epsilon", x)
  x <- gsub("pancreatic pp cell", "PP", x)
  x <- gsub("pancreatic stellate cell", "Stellate", x)
  x <- gsub("type b pancreatic cell", "Beta", x)
  x <- gsub("mhc class ii-positive cell", "MHC Class II+", x)
  # Kidney: drop "renal" / "kidney" prefixes for compactness
  x <- gsub("thick ascending limb of loop of henle epithelial cell", "Thick Ascending Limb", x)
  x <- gsub("kidney proximal tubule epithelial cell", "Proximal Tubule", x)
  x <- gsub("distal convoluted tubule epithelial cell", "Distal Convoluted Tubule", x)
  x <- gsub("renal alpha-intercalated cell", "Alpha-Intercalated", x)
  x <- gsub("renal beta-intercalated cell", "Beta-Intercalated", x)
  x <- gsub("renal principal cell", "Principal", x)
  x <- gsub("kidney leukocyte cell", "Leukocyte", x)
  x <- gsub("parietal epithelial cell", "Parietal Epithelial", x)
  x <- gsub("urothelial cell", "Urothelial", x)
  # Generic
  x <- gsub("endothelial cell", "Endothelial", x)
  x <- gsub("mast cell", "Mast", x)
  x <- gsub("erythroid lineage cell", "Erythroid", x)
  x <- gsub("b cell", "B Cell", x)
  x <- gsub("t cell", "T Cell", x)
  x <- gsub("natural killer cell", "NK Cell", x)
  x <- tools::toTitleCase(x)
  x
}


# ============================================================
# Robust legend extractor for ggplot2 3.5.x + cowplot 1.1.1
# ============================================================
extract_legend <- function(p, position = c("bottom", "right", "left", "top")) {
  position <- match.arg(position)
  gt <- ggplot2::ggplotGrob(p)

  wanted <- paste0("guide-box-", position)
  hit <- which(gt$layout$name == wanted)

  if (length(hit) == 1L) {
    g <- gt$grobs[[hit]]
    if (!inherits(g, "zeroGrob")) return(g)
  }

  old_hit <- which(gt$layout$name == "guide-box")
  if (length(old_hit) > 0L) {
    for (i in old_hit) {
      g <- gt$grobs[[i]]
      if (!inherits(g, "zeroGrob")) return(g)
    }
  }

  stop(sprintf(
    "Could not extract non-empty %s legend. Available guide boxes: %s",
    position,
    paste(gt$layout$name[grepl("^guide-box", gt$layout$name)], collapse = ", ")
  ))
}

# ============================================================
# Build one row: UMAP pred + UMAP true + legend + confusion matrix
# Returns list(pred, true, legend, conf) as individual ggplot objects
# ============================================================
build_panel_parts <- function(dataset_id, dataset_label, tissue_label, subsample_frac = 1.0) {
  umap_file <- file.path(fm_root, dataset_id, paste0(dataset_id, "_tsne.csv"))
  umap_df <- read_csv(umap_file, show_col_types = FALSE)

  # Drop any stale original-dataset UMAP columns so rename below binds to t-SNE only.
  if ("UMAP_1" %in% names(umap_df)) umap_df <- umap_df %>% select(-UMAP_1, -UMAP_2)
  if ("tsne1" %in% names(umap_df)) umap_df <- umap_df %>% rename(UMAP1 = tsne1, UMAP2 = tsne2)
  if ("umap1" %in% names(umap_df)) umap_df <- umap_df %>% rename(UMAP1 = umap1, UMAP2 = umap2)
  if ("cell_type_plot" %in% names(umap_df)) {
    umap_df <- umap_df %>% rename(truth = cell_type_plot)
  } else if ("cell_type" %in% names(umap_df)) {
    umap_df <- umap_df %>% rename(truth = cell_type)
  }

  preds_csv <- file.path(project_root, "resources", "scLEAP", "results", "tables",
                         "Fig2_zero_shot_all_cells_truth_pred_dataset.csv")
  preds_df <- read_csv(preds_csv, show_col_types = FALSE) %>%
    filter(grepl(gsub("\\.project$", "", dataset_id), dataset)) %>%
    mutate(pred = tolower(predicted_cell_type))

  if ("cell_id" %in% names(umap_df) && "cell_id" %in% names(preds_df)) {
    umap_df <- umap_df %>% left_join(preds_df %>% select(cell_id, pred), by = "cell_id")
  } else {
    umap_df$pred <- preds_df$pred[1:nrow(umap_df)]
  }

  text_df <- NULL
  if ("type" %in% names(umap_df)) {
    text_df <- umap_df %>% filter(type == "text")
    text_df$truth_clean <- clean_ct(tolower(text_df$truth))
    umap_df <- umap_df %>% filter(type == "cell")
  }

  umap_df$truth <- tolower(umap_df$truth)
  umap_df$truth_clean <- clean_ct(umap_df$truth)
  umap_df$pred_clean <- clean_ct(umap_df$pred)

  # Optional per-cell-type subsampling to reduce overplotting on dense datasets.
  if (subsample_frac < 1.0) {
    set.seed(1)
    umap_df <- umap_df %>% group_by(truth_clean) %>% sample_frac(subsample_frac) %>% ungroup()
    message(sprintf("Subsampled %s to %d cells (fraction=%.2f)",
                    dataset_id, nrow(umap_df), subsample_frac))
  }

  ct_levels <- sort(unique(umap_df$truth_clean))
  n_ct <- length(ct_levels)
  # First pin the special-story colors (if present), then fill remaining cell
  # types from the unused NPG palette entries in order.
  pins <- pinned_cell_colors[names(pinned_cell_colors) %in% ct_levels]
  remaining <- setdiff(ct_levels, names(pins))
  avail <- setdiff(npg_colors, pins)
  if (length(remaining) <= length(avail)) {
    rest <- setNames(avail[seq_along(remaining)], remaining)
  } else {
    rest <- setNames(colorRampPalette(avail)(length(remaining)), remaining)
  }
  ct_colors <- c(pins, rest)[ct_levels]


  # Dataset-specific legend layout
  # Blood: 2 rows for Cell Type, 2 rows for Type
  # Kidney: 3 rows for Cell Type, 2 rows for Type
  if (dataset_label == "GSE185948" || tissue_label == "Kidney") {

    # ==========================================================
    # Kidney
    # Cell Type: 3 rows
    # Type:      2 rows
    # ==========================================================

    celltype_rows <- 3
    type_rows <- 2

    # ----------------------------------------------------------
    # Combined UMAP panel: zeroshot_tsne_panel.pdf
    # ----------------------------------------------------------
    legend_cell_x <- 0.10
    legend_cell_y <- 0.09
    legend_cell_w <- 0.62
    legend_cell_h <- 0.15

    legend_type_x <- 0.73
    legend_type_y <- 0.09
    legend_type_w <- 0.17
    legend_type_h <- 0.14

    # ----------------------------------------------------------
    # Standalone: zeroshot_kidney_results.pdf
    # ----------------------------------------------------------

    # Cell Type legend
    stand_legend_cell_x <- 0.1
    stand_legend_cell_y <- 0.165
    stand_legend_cell_w <- 0.64
    stand_legend_cell_h <- 0.13

    # Type legend
    stand_legend_type_x <- 0.76
    stand_legend_type_y <- 0.180
    stand_legend_type_w <- 0.14
    stand_legend_type_h <- 0.11

  } else {

    # ==========================================================
    # Blood
    # Cell Type: 2 rows
    # Type:      2 rows
    # ==========================================================

    celltype_rows <- 2
    type_rows <- 2

    # ----------------------------------------------------------
    # Combined UMAP panel: zeroshot_tsne_panel.pdf
    # ----------------------------------------------------------
    legend_cell_x <- 0.16
    legend_cell_y <- 0.095
    legend_cell_w <- 0.50
    legend_cell_h <- 0.13

    # Move Type LEFT
    legend_type_x <- 0.62
    legend_type_y <- 0.095
    legend_type_w <- 0.15
    legend_type_h <- 0.12

    # ----------------------------------------------------------
    # Standalone: zeroshot_blood_results.pdf
    # ----------------------------------------------------------

    # Cell Type legend
    stand_legend_cell_x <- 0.18
    stand_legend_cell_y <- 0.185
    stand_legend_cell_w <- 0.46
    stand_legend_cell_h <- 0.11

    # Type legend — move it LEFT, close to Cell Type
    stand_legend_type_x <- 0.61
    stand_legend_type_y <- 0.185
    stand_legend_type_w <- 0.13
    stand_legend_type_h <- 0.10
  }
  umap_theme <- theme_minimal(base_size = 8) +
    theme(
      axis.line = element_line(colour = "grey50", linewidth = 0.3),
      panel.grid = element_blank(),
      legend.position = "none",
      axis.text = element_blank(), axis.ticks = element_blank(),
      axis.title.x = element_text(size = 9, face = "bold", margin = margin(t = 3)),
      axis.title.y = element_blank(),
      plot.margin = margin(2, 0, 2, 0),
      aspect.ratio = 0.9
    )

  # Metrics annotation
  metrics_csv <- file.path(project_root, "resources", "scLEAP", "results", "tables",
                           "Fig2_zero_shot_all_metrics_summary.csv")
  m_df <- read_csv(metrics_csv, show_col_types = FALSE) %>%
    mutate(dataset = gsub("\\.project$", "", dataset)) %>%
    filter(dataset == gsub("\\.project$", "", dataset_id))

  title_rich <- sprintf("**%s (%s)**", tissue_label, dataset_label)
  metrics_subtitle <- ""
  if (nrow(m_df) > 0) {
    metrics_subtitle <- sprintf("F1: %.2f   Precision: %.2f   Recall: %.2f",
                                m_df$f1_macro[1], m_df$precision_macro[1], m_df$recall_macro[1])
  }

  # UMAP Predicted
  p_pred <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = pred_clean)) +
    geom_point(size = 0.3, alpha = 0.7, shape = 16) +
    scale_color_manual(values = ct_colors) +
    scale_x_continuous(expand = c(0.02, 0)) +
    scale_y_continuous(expand = c(0.02, 0)) +
    labs(x = "scLEAP Zero-shot Annotation", y = NULL) +
    umap_theme

  if (!is.null(text_df) && nrow(text_df) > 0) {
    p_pred <- p_pred +
      geom_point(data = text_df, aes(x = UMAP1, y = UMAP2, fill = truth_clean),
                 shape = 24, size = 1.5, stroke = 0.4, color = "black", show.legend = FALSE) +
      scale_fill_manual(values = ct_colors, guide = "none")
  }

  # Use plot.title with explicit top margin for alignment with matrix title
  if (nrow(m_df) > 0) {
    plain_title <- sprintf("%s (%s), F1=%.2f", tissue_label, dataset_label, m_df$f1_macro[1])
  } else {
    plain_title <- sprintf("%s (%s)", tissue_label, dataset_label)
  }
  p_pred <- p_pred +
    labs(title = plain_title) +
    coord_cartesian(clip = "off") +
    theme(plot.title = element_text(size = 9, face = "bold", hjust = 0,
                                     margin = margin(t = 4, b = 2)),
          plot.title.position = "plot",
          plot.margin = margin(2, 0, 2, 0))

  # UMAP Ground Truth (no title — dataset title only on pred)
  p_true <- ggplot(umap_df, aes(x = UMAP1, y = UMAP2, color = truth_clean)) +
    geom_point(size = 0.3, alpha = 0.7, shape = 16) +
    scale_color_manual(values = ct_colors) +
    scale_x_continuous(expand = c(0.02, 0)) +
    scale_y_continuous(expand = c(0.02, 0)) +
    labs(x = "Ground Truth", y = NULL, title = " ") +
    umap_theme +
    coord_cartesian(clip = "off") +
    theme(plot.title = element_text(size = 9, hjust = 0, margin = margin(t = 4, b = 2)),
          plot.title.position = "plot",
          plot.margin = margin(2, 0, 2, 0))

  if (!is.null(text_df) && nrow(text_df) > 0) {
    p_true <- p_true +
      geom_point(data = text_df, aes(x = UMAP1, y = UMAP2, fill = truth_clean),
                 shape = 24, size = 1.5, stroke = 0.4, color = "black", show.legend = FALSE) +
      scale_fill_manual(values = ct_colors, guide = "none")
  }

  # ============================================================
  # Legends
  # ============================================================

  # Cell-type legend only
  p_legend_celltype <- ggplot(
    umap_df,
    aes(x = UMAP1, y = UMAP2, color = truth_clean)
  ) +
    geom_point(size = 0.3, alpha = 0.7) +
    scale_color_manual(
      values = ct_colors,
      breaks = ct_levels,
      name = "Cell Type"
    ) +
    guides(
      color = guide_legend(
        nrow = celltype_rows,
        byrow = TRUE,
        title.position = "left",
        override.aes = list(size = 4, alpha = 1, shape = 16)
      )
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.justification = "center",
      legend.text = element_text(size = 8.5),
      legend.title = element_text(size = 9, face = "bold"),
      legend.key.size = unit(0.40, "cm"),
      legend.spacing.x = unit(0.20, "cm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      legend.box.spacing = unit(0, "cm")
    )

  # cowplot 1.1.1 compatibility: no legend= argument
  legend_celltype <- extract_legend(p_legend_celltype, "bottom")

  # Type legend only
  type_df <- data.frame(
    x = c(1, 2),
    y = c(1, 1),
    Type = factor(c("Cell", "Text"), levels = c("Cell", "Text"))
  )

  p_legend_type <- ggplot(type_df, aes(x = x, y = y, shape = Type)) +
    geom_point(size = 3) +
    scale_shape_manual(
      values = c("Cell" = 16, "Text" = 24),
      name = "Type"
    ) +
    guides(
      shape = guide_legend(
        nrow = type_rows,
        byrow = TRUE,
        title.position = "left",
        override.aes = list(size = 3, fill = "grey30")
      )
    ) +
    theme_void() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.justification = "center",
      legend.text = element_text(size = 8.5),
      legend.title = element_text(size = 9, face = "bold"),
      legend.key.size = unit(0.40, "cm"),
      legend.spacing.x = unit(0.15, "cm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0),
      legend.box.spacing = unit(0, "cm")
    )

  legend_type <- extract_legend(p_legend_type, "bottom")

  # Combined horizontal legend for the UMAP-only panel
  # This is a cowplot/ggplot object, so later it must be drawn with draw_plot().
  legend_grob_horiz <- cowplot::plot_grid(
    legend_celltype,
    legend_type,
    nrow = 1,
    rel_widths = c(0.80, 0.20),
    align = "h"
  )

  # Vertical cell-type legend for standalone panels
  p_legend_vertical <- ggplot(
    umap_df,
    aes(x = UMAP1, y = UMAP2, color = truth_clean)
  ) +
    geom_point(size = 0.3, alpha = 0.7) +
    scale_color_manual(
      values = ct_colors,
      breaks = ct_levels,
      name = "Cell Type"
    ) +
    guides(
      color = guide_legend(
        ncol = 1,
        override.aes = list(size = 3, alpha = 1, shape = 16)
      )
    ) +
    theme_void() +
    theme(
      legend.position = "right",
      legend.text = element_text(size = 10),
      legend.title = element_text(size = 11, face = "bold"),
      legend.key.size = unit(0.28, "cm"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(0, 0, 0, 0)
    )

  legend_grob <- extract_legend(p_legend_vertical, "right")

  message(sprintf(
    "Legend classes for %s: celltype=%s, type=%s, vertical=%s",
    dataset_label,
    class(legend_celltype)[1],
    class(legend_type)[1],
    class(legend_grob)[1]
  ))

  # Confusion matrix
  all_ct <- sort(unique(c(umap_df$truth_clean, umap_df$pred_clean)))
  conf_raw <- umap_df %>%
    count(truth_clean, pred_clean, name = "n") %>%
    group_by(truth_clean) %>%
    mutate(frac = n / sum(n)) %>%
    ungroup()

  conf_df <- expand.grid(truth_clean = all_ct, pred_clean = all_ct, stringsAsFactors = FALSE) %>%
    left_join(conf_raw, by = c("truth_clean", "pred_clean")) %>%
    mutate(
      n = ifelse(is.na(n), 0, n),
      frac = ifelse(is.na(frac), 0, frac),
      truth_clean = factor(truth_clean, levels = rev(all_ct)),
      pred_clean = factor(pred_clean, levels = all_ct),
      text_color = ifelse(frac > 0.5, "white", "black")
    )

  p_conf <- ggplot(conf_df, aes(x = pred_clean, y = truth_clean, fill = frac)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = sprintf("%.2f", frac), color = text_color), size = 2.0, show.legend = FALSE) +
    scale_color_identity() +
    scale_fill_gradient(low = "#E8EDF2", high = "#3C5488FF", limits = c(0, 1), name = NULL) +
    scale_x_discrete(expand = c(0, 0)) +
    scale_y_discrete(expand = c(0, 0)) +
    labs(x = "Predicted Cell Type", y = "True Cell Type", title = metrics_subtitle) +
    theme_minimal(base_size = 8) +
    theme(
      aspect.ratio = 0.65,
      axis.text.x = element_text(size = 6.5, angle = 45, hjust = 1, vjust = 1,
                                  margin = margin(t = -4)),
      axis.text.y = element_text(size = 6.5, hjust = 1,
                                  margin = margin(r = -2)),
      axis.title.y = element_text(size = 8, angle = 90, margin = margin(r = 2)),
      axis.title.x = element_text(size = 8, margin = margin(t = 2)),
      axis.ticks = element_blank(),
      axis.ticks.length = unit(-2, "pt"),
      plot.title = element_text(size = 9, hjust = 1, face = "bold", margin = margin(t = 6, b = 3)),
      plot.title.position = "plot",
      legend.position = "none",
      panel.grid = element_blank(),
      plot.margin = margin(0, 0, 2, 0)
    )

  # list(pred = p_pred, true = p_true, legend = legend_grob,
  #      legend_horiz = legend_grob_horiz, conf = p_conf)
  list(
    pred = p_pred,
    true = p_true,
    legend = legend_grob,
    legend_horiz = legend_grob_horiz,
    legend_celltype = legend_celltype,
    legend_type = legend_type,
    conf = p_conf,
    legend_cell_x = legend_cell_x,
    legend_cell_y = legend_cell_y,
    legend_cell_w = legend_cell_w,
    legend_cell_h = legend_cell_h,
    legend_type_x = legend_type_x,
    legend_type_y = legend_type_y,
    legend_type_w = legend_type_w,
    legend_type_h = legend_type_h,
    stand_legend_cell_x = stand_legend_cell_x,
    stand_legend_cell_y = stand_legend_cell_y,
    stand_legend_cell_w = stand_legend_cell_w,
    stand_legend_cell_h = stand_legend_cell_h,
    stand_legend_type_x = stand_legend_type_x,
    stand_legend_type_y = stand_legend_type_y,
    stand_legend_type_w = stand_legend_type_w,
    stand_legend_type_h = stand_legend_type_h
  )
}

# ============================================================
# Build both panels
# ============================================================
message("Building Blood panel...")
bld <- build_panel_parts("E-MTAB-9221.project", "E-MTAB-9221", "Blood")

message("Building Kidney panel...")
kid <- build_panel_parts("GSE185948", "GSE185948", "Kidney", subsample_frac = 0.5)

# ============================================================
# Generate each panel independently via ggdraw with explicit per-element placement.
# Every element (pred, true, legend, conf) gets its own draw_plot/draw_grob call,
# all anchored at y=0 height=1 so all titles sit at the same top edge.
# This avoids patchwork's internal vertical centering of aspect-ratio-constrained plots.
# ============================================================
crop_pdf <- function(tmp_file, out_file) {
  system2(
    "env",
    c(
      "-u", "TEXMFROOT",
      "-u", "TEXMFHOME",
      "-u", "TEXMFCNF",
      "-u", "TEXINPUTS",
      "-u", "PERL5LIB",
      "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
      "/usr/bin/pdfcrop",
      tmp_file,
      out_file
    ),
    stdout = FALSE,
    stderr = FALSE
  )

  if (!file.exists(out_file)) {
    file.copy(tmp_file, out_file, overwrite = TRUE)
  }

  file.remove(tmp_file)
  message("Saved: ", out_file)
}

make_panel_pdf <- function(parts, out_file,
                           panel_w = 8.6,
                           panel_h = 4.5) {

  tight <- function(p) {
    p + theme(
      aspect.ratio = NULL,
      plot.margin = margin(2, -12, 0, -12),
      axis.title.x = element_text(
        size = 10,
        face = "bold",
        margin = margin(t = 2)
      ),
      plot.title = element_text(
        size = 10,
        face = "bold",
        margin = margin(l = 13, b = 1)
      )
    )
  }

  pred_plot <- tight(parts$pred)
  true_plot <- tight(parts$true)

  tmp_file <- sub("\\.pdf$", ".tmp.pdf", out_file)

  full <- ggdraw(xlim = c(0, 1), ylim = c(0, 1)) +
    draw_plot(
      pred_plot,
      x = 0.04,
      y = 0.31,
      width = 0.42,
      height = 0.66
    ) +
    draw_plot(
      true_plot,
      x = 0.49,
      y = 0.31,
      width = 0.42,
      height = 0.66
    ) +
    draw_grob(
      parts$legend_celltype,
      x = parts$stand_legend_cell_x,
      y = parts$stand_legend_cell_y,
      width = parts$stand_legend_cell_w,
      height = parts$stand_legend_cell_h
    ) +
    draw_grob(
      parts$legend_type,
      x = parts$stand_legend_type_x,
      y = parts$stand_legend_type_y,
      width = parts$stand_legend_type_w,
      height = parts$stand_legend_type_h
    )

  ggsave(
    tmp_file,
    full,
    width = panel_w,
    height = panel_h,
    device = "pdf"
  )

  crop_pdf(tmp_file, out_file)
}

c_file <- file.path(out_dir, "zeroshot_blood_results.pdf")
d_file <- file.path(out_dir, "zeroshot_kidney_results.pdf")

make_panel_pdf(bld, c_file)
make_panel_pdf(kid, d_file)

# ============================================================
# Figure 2 Panel B: combined UMAP-only composite for both datasets.
# Layout: [panc pred | panc true | panc legend] [blood pred | blood true | blood legend]
# Uses the same UMAP / legend grobs built above so styling matches the standalone
# panels. No confusion matrix and no per-UMAP legends — each dataset has one
# shared legend.
# ============================================================
make_umaps_panel_pdf <- function(parts_list, out_file,
                                 panel_w = 12,    # total width (inches)
                                 panel_h = 4.3) {
  # Per dataset: two UMAPs on top row, shared horizontal legend below (two
  # legend groups stacked vertically inside the legend grob). Strip plot
  # margins and any legend box margin so neighbouring plots touch
  # edge-to-edge.
  # Force square UMAPs (aspect.ratio = 1) and size the figure so each UMAP
  # cell is naturally square: panel_w = 12 in / 4 UMAPs = 3 in wide each;
  # rel_heights c(1, 0.42) with panel_h = 4.3 in gives a 3.03-in-tall UMAP
  # row — effectively square cells with no letterboxing.
  tight <- function(p) {
    p + theme(aspect.ratio = 1,
              plot.margin = margin(0, 0, 0, 0),
              axis.title.x = element_text(size = 9, face = "bold",
                                          margin = margin(t = 4)),
              plot.background = element_blank(),
              panel.background = element_blank())
  }
  build_block <- function(parts) {
    umap_row <- (tight(parts$pred) | tight(parts$true)) +
      plot_layout(widths = c(1, 1)) &
      theme(plot.margin = margin(0, 0, 0, 0))

    # Keep UMAPs large and reserve a compact strip below for the legends.
    umap_h <- 0.76
    umap_bottom <- 0.18

    ggdraw(xlim = c(0, 1), ylim = c(0, 1)) +
      draw_plot(
        umap_row,
        x = 0,
        y = umap_bottom,
        width = 1,
        height = umap_h
      ) +
      draw_grob(
        parts$legend_celltype,
        x = parts$legend_cell_x,
        y = parts$legend_cell_y,
        width = parts$legend_cell_w,
        height = parts$legend_cell_h
      ) +
      draw_grob(
        parts$legend_type,
        x = parts$legend_type_x,
        y = parts$legend_type_y,
        width = parts$legend_type_w,
        height = parts$legend_type_h
      )
  }

  blocks <- lapply(parts_list, build_block)
  full <- plot_grid(plotlist = blocks, nrow = 1, rel_widths = rep(1, length(blocks)))

  ggsave(out_file, full, width = panel_w, height = panel_h, device = "pdf")
  message("Saved: ", out_file)
}

umaps_file <- file.path(out_dir, "zeroshot_tsne_panel.pdf")
make_umaps_panel_pdf(list(bld, kid), umaps_file)
