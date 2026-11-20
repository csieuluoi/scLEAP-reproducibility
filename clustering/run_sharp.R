#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(RhpcBLASctl)
  library(SHARP)
  library(SingleCellExperiment)
  library(mclust)
  library(ClusterR)
  library(purrr)
  library(anndata)
  library(parallel)
  library(Matrix)
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
        out[[key]] <- "1"
        i <- i + 1
      } else if (startsWith(args[i + 1], "--")) {
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
    "  Rscript run_sharp_subsample.R --data_dir <DIR> --result_dir <DIR> ",
    "[--subsample_dir <DIR>] [--max_cells 3000000] [--seed 1] [--dataset <NAME>] ",
    "[--parallel 1] [--n_procs 4]\n"
  )
}

base_path <- args$data_dir
result_root <- args$result_dir
subsample_dir <- if (!is.null(args$subsample_dir)) args$subsample_dir else file.path(result_root, "_subsamples")

max_cells <- if (!is.null(args$max_cells)) as.integer(args$max_cells) else 300000L
seed <- if (!is.null(args$seed)) as.integer(args$seed) else 1L
only_dataset <- if (!is.null(args$dataset)) args$dataset else NA_character_

parallel_on <- !is.null(args$parallel)
n_procs <- if (!is.null(args$n_procs)) as.integer(args$n_procs) else 4L

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

process_tissue <- function(tissue) {
  tissue_out <- file.path(result_root, tissue)
  if (!dir.exists(tissue_out)) dir.create(tissue_out, recursive = TRUE)

  log_file <- file.path(tissue_out, "debug_log.txt")
  log_msg <- function(...) {
    msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", paste(..., collapse = ""))
    cat(msg, "\n", file = log_file, append = TRUE)
    message(msg)
  }

  tryCatch({
    start_time <- Sys.time()
    log_msg("[START] tissue=", tissue, " pid=", Sys.getpid())

    data_test_path <- file.path(base_path, tissue, "test.h5ad")
    if (!file.exists(data_test_path)) stop("Missing test.h5ad: ", data_test_path)
    log_msg("Found file")

    data_test <- anndata::read_h5ad(data_test_path)
    log_msg("Loaded h5ad")

    if (is.null(data_test$X)) stop("data_test$X is NULL")
    log_msg("X dim = ", paste(dim(data_test$X), collapse = " x "))

    cell_ids <- rownames(data_test$obs)
    if (is.null(cell_ids) || any(!nzchar(cell_ids))) {
      cell_ids <- paste0("Cell_", seq_len(nrow(data_test$obs)))
      rownames(data_test$obs) <- cell_ids
      log_msg("Assigned fallback cell IDs")
    }

    chosen <- get_or_create_manifest(cell_ids, tissue)
    log_msg("Chosen cells: ", length(chosen), " / ", length(cell_ids))

    if (length(chosen) < length(cell_ids)) {
      data_test <- data_test[chosen, ]
      log_msg("Subsampled AnnData")
    }

    count_matrix <- t(data_test$X)
    log_msg("Count matrix dim = ", paste(dim(count_matrix), collapse = " x "))

    cell_sums <- Matrix::colSums(count_matrix)
    log_msg("Computed cell sums")

    n_zero <- sum(cell_sums == 0)
    if (n_zero > 0) log_msg("Zero-count cells: ", n_zero)
    cell_sums[cell_sums == 0] <- 1

    count_matrix_norm <- t(t(count_matrix) / cell_sums * 1e4)
    log_msg("Normalized")

    if (!inherits(count_matrix_norm, "dgCMatrix")) {
      count_matrix_log <- as(count_matrix_norm, "dgCMatrix")
      log_msg("Converted to dgCMatrix")
    } else {
      count_matrix_log <- count_matrix_norm
    }

    if (length(count_matrix_log@x) > 0 && max(count_matrix_log@x) > 20) {
      count_matrix_log@x <- log1p(count_matrix_log@x)
      log_msg("Applied log1p")
    } else {
      log_msg("Skipped log1p")
    }

    if (!(ct_key %in% colnames(data_test$obs))) {
      stop("Missing truth column: ", ct_key)
    }
    label_df <- data_test$obs[, ct_key, drop = FALSE]
    log_msg("Labels found, unique types = ", length(unique(label_df[[ct_key]])))

    sharp_cores <- if (parallel_on) 1L else 5L
    log_msg("Running SHARP with n.cores=", sharp_cores)

    sharp_out <- purrr::quietly(SHARP)(
      count_matrix_log,
      rN.seed = 1,
      n.cores = sharp_cores,
      partition.ncells = 2000
    )

    if (length(sharp_out$warnings) > 0) {
      log_msg("SHARP warnings: ", paste(sharp_out$warnings, collapse = " | "))
    }
    if (length(sharp_out$messages) > 0) {
      log_msg("SHARP messages: ", paste(sharp_out$messages, collapse = " | "))
    }

    res <- sharp_out$result
    if (is.null(res$pred_clusters)) stop("SHARP returned NULL pred_clusters")

    cluster <- as.numeric(res$pred_clusters)
    if (length(cluster) != nrow(label_df)) {
      stop("Cluster length mismatch: ", length(cluster), " vs ", nrow(label_df))
    }
    log_msg("SHARP finished, clusters=", length(unique(cluster)))

    truth_vec <- as.numeric(factor(label_df[[ct_key]]))
    val <- sapply(
      c("adjusted_rand_index", "jaccard_index", "purity", "nmi"),
      function(x) round(external_validation(truth_vec, cluster, method = x), 2)
    )
    log_msg("Metrics computed")

    cluster_df <- data.frame(
      cell = rownames(label_df),
      true_label = as.character(label_df[[ct_key]]),
      cluster = cluster
    )

    elapsed_time <- difftime(Sys.time(), start_time, units = "secs")
    
    writeLines(sprintf(
      paste0(
        "Tissue: %s\nMethod: SHARP\nARI: %.2f\nJaccard: %.2f\n",
        "Purity: %.2f\nNMI: %.2f\nSilhouette cluster score: -1\nSilhouette: -1\n",
        "Time: %.2f\n"
      ),
      tissue, val["adjusted_rand_index"], val["jaccard_index"], val["purity"], val["nmi"], 
      as.numeric(elapsed_time)
    ), file.path(tissue_out, "results.txt"))

    write.csv(cluster_df, file.path(tissue_out, "clusters.csv"), row.names = FALSE)
    log_msg("[OK] finished in ", round(as.numeric(elapsed_time), 2), " seconds")

    TRUE
  }, error = function(e) {
    msg <- paste0("[ERR] tissue=", tissue, " | ", conditionMessage(e))
    cat(msg, "\n", file = file.path(result_root, "error_log.txt"), append = TRUE)
    cat(msg, "\n", file = log_file, append = TRUE)
    message(msg)
    FALSE
  })
}


# ---------------------------
# Main
# ---------------------------
tissue_file <- "/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/new_cellxgene_download/meta-data/valid_tissues_full_data.txt"

# Load tissues from file (one tissue per line)
tissues <- readLines(tissue_file, warn = FALSE)
tissues <- trimws(tissues)
tissues <- tissues[nzchar(tissues)]
tissues <- tissues[!startsWith(tissues, "#")]

available_tissues <- list.dirs(base_path, full.names = FALSE, recursive = FALSE)
tissues <- intersect(tissues, available_tissues)

remain_tissues <- tissues[!file.exists(file.path(result_root, tissues, "results.txt"))]
message(sprintf("Tissues to process: %d (parallel=%s, n_procs=%d)", length(remain_tissues), parallel_on, n_procs))

if (parallel_on && length(remain_tissues) > 1) {
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
# Rscript run_sharp.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results/sharp \
#   --max_cells 30000000 \
#   --seed 1
#
# Parallel (default 4 processes):
# Rscript run_sharp.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results/sharp \
#   --max_cells 3000000 \
#   --seed 1 \
#   --parallel 1
#
# Parallel with explicit n_procs:
# Rscript run_sharp.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results-with-running-time/sharp \
#   --max_cells 3000000 \
#   --seed 1 \
#   --parallel 1 --n_procs 4
