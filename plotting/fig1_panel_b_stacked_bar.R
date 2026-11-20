# Figure 1 panel B: stacked cell-type composition bars.
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(grid)
})


# =========================================================
# PATHS
# =========================================================

project_root <-
  "./"

cell_count_file <- file.path(
  project_root,
  "resources", "scLEAP", "results",
  "tables",
  "celltype_counts.csv"
)

valid_tissue_file <-
  "./resources/scLEAP/data/meta-data/valid_tissues_full_data.txt"

out_dir <- "./figure"

dir.create(
  out_dir,
  showWarnings = FALSE,
  recursive = TRUE
)


# =========================================================
# CONTROLS
# =========================================================

EXPECTED_N_TISSUES <- 294

TOP_N_TISSUES <- 15
TOP_N_CT      <- 10

SAVE_WIDTH  <- 11.5
SAVE_HEIGHT <- 7.8

BASE_FONT_SIZE     <- 24
AXIS_TEXT_SIZE     <- 20
AXIS_TITLE_SIZE    <- 24

LEGEND_TEXT_SIZE   <- 20
LEGEND_TITLE_SIZE  <- 21
TISSUE_LABEL_SIZE  <- 7.3

LEGEND_X <- 0.85
LEGEND_Y <- 0.45

TISSUE_LABEL_NUDGE <- 250000
X_RIGHT_EXPAND     <- 0.35


# =========================================================
# HELPER FUNCTIONS
# =========================================================

# Use exactly the same normalization for tissue names
# from BOTH files.

normalize_tissue <- function(x) {

  x <- as.character(x)

  x <- trimws(x)

  x <- gsub(
    "_",
    " ",
    x
  )

  x <- tools::toTitleCase(
    tolower(x)
  )

  x
}


format_count <- function(n) {

  ifelse(
    n >= 1e6,
    sprintf(
      "%.1fM",
      n / 1e6
    ),
    sprintf(
      "%.0fK",
      n / 1e3
    )
  )
}


# =========================================================
# LOAD VALID 294 TISSUES
# =========================================================

valid_tissues <- readLines(
  valid_tissue_file,
  warn = FALSE
)

valid_tissues <- trimws(
  valid_tissues
)

# Remove empty lines
valid_tissues <- valid_tissues[
  nzchar(valid_tissues)
]


# If the text file accidentally contains a simple
# "tissue" / "tissues" header, remove it.

if (
  length(valid_tissues) > 0 &&
  tolower(valid_tissues[1]) %in% c(
    "tissue",
    "tissues"
  )
) {

  valid_tissues <- valid_tissues[-1]
}


# Normalize names
valid_tissues <- normalize_tissue(
  valid_tissues
)


# Remove accidental duplicated lines
valid_tissues <- unique(
  valid_tissues
)


message(
  "=========================================="
)

message(
  "VALID TISSUE FILE"
)

message(
  "Number of unique tissues in file: ",
  length(valid_tissues)
)


# Strong check: this file is expected to define 294 tissues.

if (
  length(valid_tissues) != EXPECTED_N_TISSUES
) {

  stop(
    paste0(
      "Expected ",
      EXPECTED_N_TISSUES,
      " unique tissues in valid_tissues_full_data.txt, ",
      "but found ",
      length(valid_tissues),
      "."
    )
  )
}


# =========================================================
# LOAD CELL-TYPE COUNTS
# =========================================================

df_raw <- read.csv(
  cell_count_file,
  stringsAsFactors = FALSE
)


# =========================================================
# CLEAN BASIC FIELDS
# =========================================================

df_raw$tissue <- as.character(
  df_raw$tissue
)

df_raw$cell_type <- as.character(
  df_raw$cell_type
)


# Missing tissue labels cannot be matched to the valid list.
df_raw$tissue <- trimws(
  df_raw$tissue
)


# Missing cell-type labels are retained as Unknown.

df_raw$cell_type[
  is.na(df_raw$cell_type) |
    trimws(df_raw$cell_type) == ""
] <- "Unknown"


# Normalize tissue names using exactly the same function
# as the 294-tissue list.

df_raw$tissue <- normalize_tissue(
  df_raw$tissue
)


# =========================================================
# CHECK TISSUE MATCHING
# =========================================================

available_tissues <- unique(
  df_raw$tissue
)


missing_valid_tissues <- setdiff(
  valid_tissues,
  available_tissues
)


if (
  length(missing_valid_tissues) > 0
) {

  message(
    "WARNING: Valid tissues not found in celltype_counts.csv:"
  )

  for (
    tissue_name in missing_valid_tissues
  ) {

    message(
      "  - ",
      tissue_name
    )
  }

  stop(
    paste0(
      length(missing_valid_tissues),
      " of the 294 requested tissues were not found ",
      "in celltype_counts.csv. Check tissue-name formatting."
    )
  )
}


# =========================================================
# FILTER TO EXACTLY THE 294 VALID TISSUES
# =========================================================

df <- df_raw %>%
  filter(
    tissue %in% valid_tissues
  )


matched_tissues <- n_distinct(
  df$tissue
)


message(
  "Matched tissues: ",
  matched_tissues
)


if (
  matched_tissues != EXPECTED_N_TISSUES
) {

  stop(
    paste0(
      "Expected ",
      EXPECTED_N_TISSUES,
      " matched tissues, but found ",
      matched_tissues,
      "."
    )
  )
}


# =========================================================
# AGGREGATE TISSUE × CELL TYPE COUNTS
# =========================================================

# IMPORTANT:
#
# Unknown and Unlabeled cells are NOT removed.
# They remain part of the full dataset.

df <- df %>%

  group_by(
    tissue,
    cell_type
  ) %>%

  summarise(
    n_cells = sum(
      number_of_cells,
      na.rm = TRUE
    ),
    .groups = "drop"
  )


# =========================================================
# FORMAT CELL-TYPE NAMES
# =========================================================

df$cell_type <- tools::toTitleCase(
  tolower(df$cell_type)
)


# Fix specific capitalization

df$cell_type <- gsub(
  "^b Cell$",
  "B Cell",
  df$cell_type
)

df$cell_type <- gsub(
  "^t Cell$",
  "T Cell",
  df$cell_type
)


# These labels remain in the dataset,
# but they cannot receive an individual legend color.

UNKNOWN_CELLTYPE_LABELS <- c(
  "Unknown",
  "Unlabeled"
)


# =========================================================
# RE-AGGREGATE AFTER NAME NORMALIZATION
# =========================================================

# Formatting could theoretically merge labels differing only
# in capitalization, so aggregate once more.

df <- df %>%

  group_by(
    tissue,
    cell_type
  ) %>%

  summarise(
    n_cells = sum(n_cells),
    .groups = "drop"
  )


# =========================================================
# WHOLE 294-TISSUE DATASET SUMMARY
# =========================================================

total_cells <- sum(
  df$n_cells
)

total_tissues <- n_distinct(
  df$tissue
)

total_cell_types <- n_distinct(
  df$cell_type
)


message(
  "=========================================="
)

message(
  "294-TISSUE DATASET"
)

message(
  "Tissues: ",
  total_tissues
)

message(
  "Cell-type labels: ",
  total_cell_types
)

message(
  "Total cells: ",
  format(
    total_cells,
    big.mark = ","
  )
)

message(
  "Total cells formatted: ",
  format_count(total_cells)
)

message(
  "=========================================="
)


# =========================================================
# TISSUE TOTALS ACROSS ALL 294 TISSUES
# =========================================================

tissue_totals <- df %>%

  group_by(
    tissue
  ) %>%

  summarise(
    total = sum(n_cells),
    .groups = "drop"
  ) %>%

  arrange(
    desc(total)
  )


# =========================================================
# TOP 15 TISSUES
# =========================================================

top_tissues <- tissue_totals %>%

  slice_head(
    n = TOP_N_TISSUES
  ) %>%

  pull(
    tissue
  )


# =========================================================
# OTHER TISSUES
# =========================================================

other_tissue_df <- tissue_totals %>%

  filter(
    !tissue %in% top_tissues
  )


n_other_tissues <- nrow(
  other_tissue_df
)


n_other_tissue_cells <- sum(
  other_tissue_df$total
)


other_tissue_label <- paste0(
  n_other_tissues,
  " Others (",
  format_count(
    n_other_tissue_cells
  ),
  ")"
)


message(
  "Top tissues displayed: ",
  length(top_tissues)
)

message(
  "Other tissues: ",
  other_tissue_label
)


# With exactly 294 tissues:
#
# 294 - 15 = 279
#
# so this should print:
# 279 Others (...)


# =========================================================
# GLOBAL CELL-TYPE TOTALS
#
# IMPORTANT:
# calculated over ALL 294 selected tissues
# =========================================================

celltype_totals <- df %>%

  group_by(
    cell_type
  ) %>%

  summarise(
    total = sum(n_cells),
    .groups = "drop"
  ) %>%

  arrange(
    desc(total)
  )


# =========================================================
# TOP 10 CELL TYPES
# =========================================================

# Unknown / Unlabeled remain in the cell counts,
# but are excluded from consideration for an individual color.

top_ct_df <- celltype_totals %>%

  filter(
    !cell_type %in%
      UNKNOWN_CELLTYPE_LABELS
  ) %>%

  slice_head(
    n = TOP_N_CT
  )


top_ct <- top_ct_df$cell_type


# =========================================================
# GLOBAL OTHER CELL TYPES
# =========================================================

# Everything other than the 10 displayed cell types belongs
# to the global Other category.
#
# This includes Unknown and Unlabeled.

other_ct_df <- celltype_totals %>%

  filter(
    !cell_type %in% top_ct
  )


n_other_cell_types <- nrow(
  other_ct_df
)


n_other_ct_cells <- sum(
  other_ct_df$total
)


other_ct_label <- paste0(
  n_other_cell_types,
  " Others (",
  format_count(
    n_other_ct_cells
  ),
  ")"
)


message(
  "------------------------------------------"
)

message(
  "Top 10 global cell types:"
)


for (
  i in seq_len(
    nrow(top_ct_df)
  )
) {

  message(
    "  ",
    i,
    ". ",
    top_ct_df$cell_type[i],
    " = ",
    format_count(
      top_ct_df$total[i]
    )
  )
}


message(
  "Other cell types: ",
  other_ct_label
)

message(
  "------------------------------------------"
)


# =========================================================
# VERIFY CELL-TYPE ACCOUNTING
# =========================================================

top_ct_cells <- sum(
  top_ct_df$total
)


celltype_reconstructed_total <-
  top_ct_cells +
  n_other_ct_cells


if (
  celltype_reconstructed_total != total_cells
) {

  stop(
    paste0(
      "Cell-type accounting failed. Difference = ",
      celltype_reconstructed_total -
        total_cells
    )
  )
}


# =========================================================
# DATA FOR THE 15 DISPLAYED TISSUES
# =========================================================

# The same GLOBAL top 10 cell types are used in every tissue.
#
# Everything else is gray Other.

df_bar <- df %>%

  filter(
    tissue %in% top_tissues
  ) %>%

  mutate(

    cell_type_display = ifelse(

      cell_type %in% top_ct,

      cell_type,

      "Other"
    )
  ) %>%

  group_by(
    tissue,
    cell_type_display
  ) %>%

  summarise(
    n_cells = sum(n_cells),
    .groups = "drop"
  )


# =========================================================
# ABBREVIATE LONG CELL-TYPE NAMES
# =========================================================

abbreviate_ct <- function(name) {

  name <- gsub(
    "L2/3-6 Intratelencephalic Projecting Glutamatergic Neuron",
    "IT Glutamatergic Neuron",
    name,
    ignore.case = TRUE
  )

  name <- gsub(
    "Cd4-Positive, Alpha-Beta t Cell",
    "CD4+ T Cell",
    name,
    ignore.case = TRUE
  )

  name <- gsub(
    "Cd8-Positive, Alpha-Beta t Cell",
    "CD8+ T Cell",
    name,
    ignore.case = TRUE
  )

  name <- gsub(
    "Central Memory Cd4-Positive, Alpha-Beta t Cell",
    "Central Memory CD4+ T Cell",
    name,
    ignore.case = TRUE
  )

  name <- gsub(
    "Naive Thymus-Derived Cd4-Positive, Alpha-Beta t Cell",
    "Naive CD4+ T Cell",
    name,
    ignore.case = TRUE
  )

  name <- gsub(
    "Luminal Adaptive Secretory Precursor Cell of Mammary Gland",
    "Luminal Secretory Precursor",
    name,
    ignore.case = TRUE
  )

  name <- gsub(
    "Fibroblast of Mammary Gland",
    "Mammary Fibroblast",
    name,
    ignore.case = TRUE
  )

  name
}


# =========================================================
# LEGEND LABELS
#
# Counts are GLOBAL across all 294 tissues.
# =========================================================

legend_names <- abbreviate_ct(
  top_ct_df$cell_type
)


legend_names <- gsub(
  "^IT Glutamatergic Neuron$",
  "IT Glutamatergic\nNeuron",
  legend_names
)

legend_names <- gsub(
  "^Glutamatergic Neuron$",
  "Glutamatergic\nNeuron",
  legend_names
)

legend_names <- gsub(
  "^Classical Monocyte$",
  "Classical\nMonocyte",
  legend_names
)

legend_names <- gsub(
  "^Natural Killer Cell$",
  "Natural Killer\nCell",
  legend_names
)

legend_names <- gsub(
  "^Retinal Rod Cell$",
  "Retinal Rod\nCell",
  legend_names
)


legend_labels <- setNames(

  paste0(
    legend_names,
    " (",
    format_count(
      top_ct_df$total
    ),
    ")"
  ),

  top_ct_df$cell_type
)


legend_labels["Other"] <-
  other_ct_label


# =========================================================
# CELL-TYPE ORDER
# =========================================================

ct_levels <- c(
  top_ct,
  "Other"
)


df_bar$cell_type_display <- factor(
  df_bar$cell_type_display,
  levels = ct_levels
)


# =========================================================
# TISSUE ORDER
# =========================================================

OTHER_TISSUE_ROW <-
  "__OTHER_TISSUES__"


# ggplot puts the first factor at the bottom.
#
# Result:
#
# Blood
# Breast
# ...
# 15th tissue
# 279 Others

tissue_levels <- c(
  OTHER_TISSUE_ROW,
  rev(top_tissues)
)


df_bar$tissue <- factor(
  df_bar$tissue,
  levels = tissue_levels
)


# =========================================================
# COLORS
# =========================================================

ct_colors <- c(

  setNames(

    colorRampPalette(
      c(
        "#E64B35",
        "#4DBBD5",
        "#00A087",
        "#3C5488",
        "#F39B7F",
        "#8491B4",
        "#91D1C2",
        "#DC0000",
        "#7E6148",
        "#B09C85"
      )
    )(
      length(top_ct)
    ),

    top_ct
  ),

  "Other" = "grey80"
)


# =========================================================
# LABELS FOR TOP 15 TISSUES
# =========================================================

total_labels <- df_bar %>%

  group_by(
    tissue
  ) %>%

  summarise(
    total = sum(n_cells),
    .groups = "drop"
  ) %>%

  mutate(

    count_label = format_count(
      total
    ),

    label_text = paste0(
      as.character(tissue),
      " (",
      count_label,
      ")"
    )
  )


total_labels$tissue <- factor(
  total_labels$tissue,
  levels = tissue_levels
)


# =========================================================
# LABEL FOR OTHER 279 TISSUES
# =========================================================

other_tissue_text <- data.frame(

  tissue = factor(
    OTHER_TISSUE_ROW,
    levels = tissue_levels
  ),

  label_text =
    other_tissue_label
)


# =========================================================
# PLOT
# =========================================================

p <- ggplot(

  df_bar,

  aes(
    x = n_cells,
    y = tissue,
    fill = cell_type_display
  )

) +

  geom_bar(
    stat = "identity",
    width = 0.8,
    colour = "black",
    linewidth = 0.15
  ) +


  # Top-15 tissue labels
  geom_text(

    data = total_labels,

    aes(
      x = total,
      y = tissue,
      label = label_text
    ),

    hjust = 0,

    nudge_x =
      TISSUE_LABEL_NUDGE,

    size =
      TISSUE_LABEL_SIZE,

    lineheight = 0.9,

    inherit.aes = FALSE
  ) +


  # Remaining 279 tissues
  geom_text(

    data =
      other_tissue_text,

    aes(
      x = 0,
      y = tissue,
      label = label_text
    ),

    hjust = 0,

    vjust = 0.5,

    size =
      TISSUE_LABEL_SIZE,

    lineheight = 0.9,

    inherit.aes = FALSE
  ) +


  scale_x_continuous(

    labels = label_number(
      scale_cut =
        cut_short_scale()
    ),

    expand = expansion(
      mult = c(
        0,
        X_RIGHT_EXPAND
      )
    ),

    breaks = c(
      0,
      4e6,
      8e6,
      12e6
    )
  ) +


  scale_y_discrete(
    drop = FALSE
  ) +


  coord_cartesian(
    clip = "off"
  ) +


  scale_fill_manual(

    values =
      ct_colors,

    breaks =
      ct_levels,

    limits =
      ct_levels,

    labels =
      legend_labels,

    drop = FALSE,

    name =
      "Cell type"
  ) +


  guides(

    fill =
      guide_legend(

        byrow = TRUE,

        ncol = 1,

        keyheight =
          unit(
            0.62,
            "cm"
          )
      )
  ) +


  labs(
    x = "Number of cells",
    y = "Tissue"
  ) +


  theme_minimal(
    base_size =
      BASE_FONT_SIZE
  ) +


  theme(

    axis.text.y =
      element_blank(),

    axis.ticks.y =
      element_blank(),


    axis.title.y =
      element_text(
        size =
          AXIS_TITLE_SIZE,
        face =
          "bold",
        margin =
          margin(r = 12)
      ),


    axis.text.x =
      element_text(
        size =
          AXIS_TEXT_SIZE
      ),


    axis.title.x =
      element_text(
        size =
          AXIS_TITLE_SIZE
      ),


    legend.position =
      c(
        LEGEND_X,
        LEGEND_Y
      ),


    legend.background =
      element_rect(
        fill = "white",
        colour = NA
      ),


    legend.text =
      element_text(
        size =
          LEGEND_TEXT_SIZE,
        lineheight = 0.9
      ),


    legend.title =
      element_text(
        size =
          LEGEND_TITLE_SIZE,
        face =
          "bold"
      ),


    legend.key.size =
      unit(
        0.45,
        "cm"
      ),


    legend.key.spacing.y =
      unit(
        11,
        "pt"
      ),


    legend.margin =
      margin(
        4,
        6,
        4,
        6
      ),


    panel.grid.major.y =
      element_blank(),

    panel.grid.minor =
      element_blank(),


    plot.margin =
      margin(
        5,
        5,
        5,
        15
      )
  )


# =========================================================
# FINAL CONSISTENCY CHECK
# =========================================================

displayed_tissue_cells <- tissue_totals %>%

  filter(
    tissue %in%
      top_tissues
  ) %>%

  summarise(
    n = sum(total)
  ) %>%

  pull(n)


tissue_reconstructed_total <-
  displayed_tissue_cells +
  n_other_tissue_cells


message(
  "=========================================="
)

message(
  "FINAL CHECKS"
)

message(
  "Selected tissues: ",
  total_tissues
)

message(
  "Total cells in 294 tissues: ",
  format(
    total_cells,
    big.mark = ","
  )
)

message(
  "Top 15 + other tissues: ",
  format(
    tissue_reconstructed_total,
    big.mark = ","
  )
)

message(
  "Top 10 + other cell types: ",
  format(
    celltype_reconstructed_total,
    big.mark = ","
  )
)

message(
  "Tissue difference: ",
  tissue_reconstructed_total -
    total_cells
)

message(
  "Cell-type difference: ",
  celltype_reconstructed_total -
    total_cells
)

message(
  "Other tissue count: ",
  n_other_tissues,
  " (expected ",
  EXPECTED_N_TISSUES -
    TOP_N_TISSUES,
  ")"
)

message(
  "=========================================="
)


# =========================================================
# SAVE
# =========================================================

tmp_file <- file.path(
  out_dir,
  "stacked_bar_tissue_celltype.tmp.pdf"
)

out_file <- file.path(
  out_dir,
  "stacked_bar_tissue_celltype.pdf"
)


ggsave(
  tmp_file,
  p,
  width = SAVE_WIDTH,
  height = SAVE_HEIGHT,
  device = "pdf"
)


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

  file.copy(
    tmp_file,
    out_file,
    overwrite = TRUE
  )
}


file.remove(
  tmp_file
)


message(
  "Saved: ",
  normalizePath(
    out_file,
    mustWork = FALSE
  )
)