#!/usr/bin/env Rscript

library(future)

future::plan("sequential")
options(future.globals.maxSize = 8 * 1024^3)  # 8 GiB

suppressPackageStartupMessages({
  library(RhpcBLASctl)
  library(Seurat)
  library(anndata)
  library(Matrix)
  library(mclust)
  library(parallel)
  library(aricode)
  library(digest)
})

RhpcBLASctl::blas_set_num_threads(1)
RhpcBLASctl::omp_set_num_threads(1)

Sys.setenv(
  OMP_NUM_THREADS = 1,
  OPENBLAS_NUM_THREADS = 1,
  MKL_NUM_THREADS = 1,
  VECLIB_MAXIMUM_THREADS = 1,
  NUMEXPR_NUM_THREADS = 1
)

ct_key <- "cell_type"

# ---------------------------
# Minimal CLI parsing: --key value  (supports flags like --parallel 1)
# ---------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  i <- 1
  while (i <= length(args)) {
    if (startsWith(args[i], "--")) {
      key <- sub("^--", "", args[i])
      if (i == length(args)) {
        # treat as flag (boolean)
        out[[key]] <- "1"
        i <- i + 1
      } else if (startsWith(args[i + 1], "--")) {
        # treat as flag (boolean)
        out[[key]] <- "1"
        i <- i + 1
      } else {
        out[[key]] <- args[i + 1]
        i <- i + 2
      }
    } else {
      i <- i + 1
    }
  }
  out
}

args <- parse_args()

if (is.null(args$data_dir) || is.null(args$result_dir)) {
  stop(
    "Usage:\n",
    "  Rscript run_seurat_subsample_mp.R --data_dir <DIR> --result_dir <DIR> ",
    "[--subsample_dir <DIR>] [--max_cells 3000000] [--seed 1] [--dataset <NAME>] ",
    "[--parallel 1] [--n_procs 4] [--mp_start fork]\n"
  )
}

base_path    <- args$data_dir
result_root  <- args$result_dir
subsample_dir <- if (!is.null(args$subsample_dir)) args$subsample_dir else file.path(result_root, "_subsamples")

max_cells <- if (!is.null(args$max_cells)) as.integer(args$max_cells) else 300000L
seed      <- if (!is.null(args$seed)) as.integer(args$seed) else 1L
only_dataset <- if (!is.null(args$dataset)) args$dataset else NA_character_

parallel_on <- !is.null(args$parallel)
n_procs <- if (!is.null(args$n_procs)) as.integer(args$n_procs) else 4L

# mp_start is informational; this script uses mclapply (fork) on Linux
mp_start <- if (!is.null(args$mp_start)) args$mp_start else "fork"

if (!dir.exists(result_root)) dir.create(result_root, recursive = TRUE)
if (!dir.exists(subsample_dir)) dir.create(subsample_dir, recursive = TRUE)

set.seed(seed)

# ---------------------------
# Deterministic SHA1 (match Python: sha1(seed:cell_id))
# ---------------------------
sha1_hex <- function(s) {
  digest::digest(s, algo = "sha1", serialize = FALSE)
}

get_or_create_manifest <- function(cell_ids, dataset) {
  manifest_path <- file.path(subsample_dir, paste0(dataset, "__max", max_cells, "__seed", seed, ".txt"))

  if (file.exists(manifest_path)) {
    ids <- readLines(manifest_path, warn = FALSE)
    ids <- ids[nzchar(ids)]
    present <- ids[ids %in% cell_ids]
    if (length(present) == 0) stop(paste("Manifest exists but no IDs matched:", manifest_path))
    return(present)
  }

  if (max_cells <= 0 || length(cell_ids) <= max_cells) {
    chosen <- cell_ids
  } else {
    keys <- vapply(cell_ids, function(cid) sha1_hex(paste0(seed, ":", cid)), character(1))
    ord <- order(keys)
    chosen <- cell_ids[ord[seq_len(max_cells)]]
  }

  writeLines(chosen, manifest_path, sep = "\n")
  chosen
}

# ---------------------------
# One dataset/tissue
# ---------------------------
process_tissue <- function(tissue) {
  tryCatch({
    message(sprintf("[RUN] tissue=%s pid=%s", tissue, Sys.getpid()))
    set.seed(seed)

    start_time <- Sys.time()

    data_test_path <- file.path(base_path, tissue, "test.h5ad")
    data_test <- anndata::read_h5ad(data_test_path)

    # ensure obs rownames exist (cell ids)
    cell_ids <- rownames(data_test$obs)
    if (is.null(cell_ids) || any(!nzchar(cell_ids))) {
      cell_ids <- paste0("Cell_", seq_len(nrow(data_test$obs)))
      rownames(data_test$obs) <- cell_ids
    }

    # deterministic subsample (shared with Python if same naming + hash rule)
    chosen <- get_or_create_manifest(cell_ids, tissue)
    if (length(chosen) < length(cell_ids)) {
      data_test <- data_test[chosen, ]
    }

    # AnnData X: cells x genes, Seurat counts: genes x cells
    count_matrix <- t(data_test$X)

    # name safety
    if (is.null(rownames(count_matrix))) rownames(count_matrix) <- paste0("Gene_", seq_len(nrow(count_matrix)))
    if (is.null(colnames(count_matrix))) colnames(count_matrix) <- paste0("Cell_", seq_len(ncol(count_matrix)))

    seurat_obj <- CreateSeuratObject(counts = count_matrix, meta.data = data_test$obs)

    # Seurat pipeline (default-ish)
    seurat_obj <- NormalizeData(seurat_obj, verbose = FALSE)
    seurat_obj <- FindVariableFeatures(seurat_obj, verbose = FALSE)
    seurat_obj <- ScaleData(seurat_obj, verbose = FALSE)

    npcs <- if (ncol(seurat_obj) > 100) 50 else 20
    seurat_obj <- RunPCA(
      seurat_obj,
      features = VariableFeatures(object = seurat_obj),
      npcs = npcs,
      verbose = FALSE
    )

    seurat_obj <- FindNeighbors(seurat_obj, dims = 1:npcs, verbose = FALSE)
    seurat_obj <- FindClusters(seurat_obj, verbose = FALSE)

    cluster_assignments <- Idents(seurat_obj)

    if (!(ct_key %in% colnames(seurat_obj@meta.data))) {
      stop(sprintf("Missing truth column '%s' in meta.data for tissue=%s", ct_key, tissue))
    }
    true_labels <- seurat_obj@meta.data[[ct_key]]

    cluster_df <- data.frame(
      cell = names(cluster_assignments),
      cluster = as.character(cluster_assignments),
      label = as.character(true_labels)
    )

    dir_path <- file.path(result_root, tissue)
    if (!dir.exists(dir_path)) dir.create(dir_path, recursive = TRUE)

    cluster_csv_file <- file.path(dir_path, "clusters.csv")
    write.csv(cluster_df, cluster_csv_file, row.names = FALSE)

    # Metrics
    ari_score <- mclust::adjustedRandIndex(cluster_assignments, true_labels)
    nmi_score <- aricode::NMI(cluster_assignments, true_labels)
    ami_score <- aricode::AMI(cluster_assignments, true_labels)
    num_clusters <- length(unique(cluster_assignments))

    # Keep silhouettes as placeholders (same as your original)
    silhouette_true <- -1
    silhouette_cluster <- -1

    end_time <- Sys.time()
    elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

    results <- sprintf(
      paste0(
        "Tissue: %s\n",
        "Method: seurat_default\n",
        "Number of clusters: %d\n",
        "ARI: %.16f\n",
        "NMI: %.16f\n",
        "AMI: %.16f\n",
        "Silhouette_true_labels: %.17f\n",
        "Silhouette_clusters: %.17f\n",
        "Time: %.2f\n"
      ),
      tissue, num_clusters, ari_score, nmi_score, ami_score, silhouette_true, silhouette_cluster, elapsed_time
    )

    output_file <- file.path(dir_path, "results.txt")
    writeLines(results, output_file)

    message(sprintf("[OK] %s -> %s", tissue, output_file))
    invisible(TRUE)

  }, error = function(e) {
    msg <- sprintf("Error processing tissue %s: %s", tissue, e$message)
    message(paste0("[ERR] ", msg))
    write(msg, file = file.path(result_root, "error_log.txt"), append = TRUE)
    invisible(FALSE)
  })
}

# ---------------------------
# Main
# ---------------------------
# tissues <- list.dirs(base_path, full.names = FALSE, recursive = FALSE)
tissue_file <- "/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/new_cellxgene_download/meta-data/valid_tissues_full_data.txt"

# Load tissues from file (one tissue per line)
tissues <- readLines(tissue_file, warn = FALSE)
tissues <- trimws(tissues)
tissues <- tissues[nzchar(tissues)]                 # drop empty lines
tissues <- tissues[!startsWith(tissues, "#")]       # optional: drop comments

# Optional filter (same behavior as before)
if (!is.na(only_dataset)) tissues <- tissues[tissues == only_dataset]

remain_tissues <- tissues[!file.exists(file.path(result_root, tissues, "results.txt"))]
message(sprintf("Tissues to process: %d (parallel=%s, n_procs=%d, mp_start=%s)",
                length(remain_tissues), parallel_on, n_procs, mp_start))

if (parallel_on && length(remain_tissues) > 1) {
  # mclapply is for Linux/macOS; on Windows it effectively runs sequentially.
  n_use <- max(1L, min(n_procs, length(remain_tissues)))
  parallel::mclapply(remain_tissues, process_tissue, mc.cores = n_use)
} else {
  for (tissue in remain_tissues) {
    process_tissue(tissue)
  }
}

message("DONE")


# Examples:
# Sequential:
# Rscript run_seurat.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results-with-running-time/seurat \
#   --max_cells 300000 \
#   --seed 1
#
# Parallel (default 4 processes):
# Rscript run_seurat.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results-with-running-time/seurat \
#   --max_cells 300000 \
#   --seed 1 \
#   --parallel 1

# Parallel with explicit n_procs:
# Rscript run_seurat.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results-with-running-time/seurat \
#   --max_cells 30000000 \
#   --seed 1 \
#   --parallel 1 --n_procs 4
