#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(RhpcBLASctl)
  library(monocle3)
  library(anndata)
  library(Matrix)
  library(parallel)
  library(mclust)
  library(aricode)
  library(digest)
})

# -----------------------------
# Hard limit threads
# -----------------------------
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

# -----------------------------
# CLI args
# -----------------------------
parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  out <- list()
  i <- 1
  while (i <= length(args)) {
    if (startsWith(args[i], "--")) {
      key <- sub("^--", "", args[i])
      if (i == length(args) || startsWith(args[i + 1], "--")) {
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
  stop("Usage: Rscript run_monocle3_subsample_mp.R --data_dir <DIR> --result_dir <DIR> [--subsample_dir <DIR>] [--max_cells 3000000] [--seed 1] [--dataset <NAME>] [--parallel 1] [--n_procs 4]")
}

base_path <- args$data_dir
result_root <- args$result_dir
subsample_dir <- if (!is.null(args$subsample_dir)) args$subsample_dir else file.path(result_root, "_subsamples")
max_cells <- if (!is.null(args$max_cells)) as.integer(args$max_cells) else 300000L
seed <- if (!is.null(args$seed)) as.integer(args$seed) else 1L
only_dataset <- if (!is.null(args$dataset)) args$dataset else NA_character_
parallel_on <- !is.null(args$parallel)
n_procs <- if (!is.null(args$n_procs)) as.integer(args$n_procs) else 4L

dir.create(result_root, recursive = TRUE, showWarnings = FALSE)
dir.create(subsample_dir, recursive = TRUE, showWarnings = FALSE)
set.seed(seed)

sha1_hex <- function(s) digest::digest(s, algo = "sha1", serialize = FALSE)

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
    chosen <- cell_ids[order(keys)[seq_len(max_cells)]]
  }

  writeLines(chosen, manifest_path)
  chosen
}

# -----------------------------
# Helpers: robust sparse conversion
# -----------------------------
to_dgCMatrix <- function(x) {
  # Convert various Matrix/dense types to dgCMatrix robustly.
  if (inherits(x, "dgCMatrix")) return(x)

  # If it's any sparse Matrix class, coerce via triplet or CsparseMatrix
  if (inherits(x, "sparseMatrix") || inherits(x, "Matrix")) {
    # dgRMatrix cannot directly coerce to dgCMatrix; route via TsparseMatrix/CsparseMatrix
    if (inherits(x, "dgRMatrix")) {
      # dgR -> dgT -> dgC
      x <- as(x, "TsparseMatrix")
      x <- as(x, "CsparseMatrix")
      return(as(x, "dgCMatrix"))
    }
    if (inherits(x, "dgTMatrix")) {
      x <- as(x, "CsparseMatrix")
      return(as(x, "dgCMatrix"))
    }
    # other Matrix types (including CsparseMatrix subclasses)
    x <- as(x, "CsparseMatrix")
    return(as(x, "dgCMatrix"))
  }

  # Dense base matrix/data.frame
  if (is.data.frame(x)) x <- as.matrix(x)
  if (is.matrix(x)) {
    return(as(Matrix(x, sparse = TRUE), "dgCMatrix"))
  }

  stop(sprintf("Unsupported X type for conversion: %s", paste(class(x), collapse = ",")))
}

extract_counts_genes_by_cells <- function(ad) {
  # Your data: X is raw counts; layers/raw not available.
  counts_cells_by_genes <- ad$X
  source <- "X"

  # Ensure cells x genes dgCMatrix
  counts_cells_by_genes <- to_dgCMatrix(counts_cells_by_genes)

  # Transpose to genes x cells for Monocle3
  counts_genes_by_cells <- t(counts_cells_by_genes)
  counts_genes_by_cells <- to_dgCMatrix(counts_genes_by_cells)

  list(counts = counts_genes_by_cells, source = source)
}

# -----------------------------
# Main per-tissue
# -----------------------------
process_tissue <- function(tissue) {
  tryCatch({
    start_time <- Sys.time()
    message(sprintf("[RUN] tissue=%s pid=%s", tissue, Sys.getpid()))

    h5 <- file.path(base_path, tissue, "test.h5ad")
    if (!file.exists(h5)) stop(sprintf("Missing file: %s", h5))

    ad <- anndata::read_h5ad(h5)

    # cell ids from obs rownames
    cell_ids <- rownames(ad$obs)
    if (is.null(cell_ids) || any(!nzchar(cell_ids))) {
      cell_ids <- paste0("Cell_", seq_len(nrow(ad$obs)))
      rownames(ad$obs) <- cell_ids
    }

    # manifest + subset
    chosen <- get_or_create_manifest(cell_ids, tissue)
    if (length(chosen) < length(cell_ids)) {
      # subset obs + X consistently via anndata slicing
      ad <- ad[chosen, ]
    }

    if (!(ct_key %in% colnames(ad$obs))) {
      stop(sprintf("Missing truth column '%s' in obs for tissue=%s", ct_key, tissue))
    }

    # Extract counts (genes x cells)
    ec <- extract_counts_genes_by_cells(ad)
    counts <- ec$counts
    counts_source <- ec$source

    # Gene names: prefer var rownames if present and matches
    gene_names <- NULL
    try({
      vn <- rownames(ad$var)
      if (!is.null(vn) && length(vn) == nrow(counts) && all(nzchar(vn))) gene_names <- vn
    }, silent = TRUE)

    if (is.null(gene_names)) {
      rn <- rownames(counts)
      if (!is.null(rn) && length(rn) == nrow(counts) && all(nzchar(rn))) gene_names <- rn
    }

    if (is.null(gene_names)) {
      gene_names <- paste0("Gene_", seq_len(nrow(counts)))
      rownames(counts) <- gene_names
    } else {
      rownames(counts) <- gene_names
    }

    gene_md <- data.frame(gene_short_name = gene_names, stringsAsFactors = FALSE)
    rownames(gene_md) <- gene_names

    # Cell metadata
    cell_md <- as.data.frame(ad$obs, stringsAsFactors = FALSE)
    cell_md$cell_id <- rownames(cell_md)

    # Construct CDS
    cds <- new_cell_data_set(
      expression_data = counts,
      cell_metadata = cell_md,
      gene_metadata = gene_md
    )

    # Monocle3 clustering pipeline
    # (preprocess_cds default num_dim=50, but keeping explicit for determinism/consistency)
    cds <- preprocess_cds(cds, num_dim = 50)
    cds <- reduce_dimension(cds, reduction_method = "UMAP")
    cds <- cluster_cells(cds, reduction_method = "UMAP")

    # Extract clusters (named by cell)
    cl_fac <- clusters(cds, reduction_method = "UMAP")
    cl <- as.character(cl_fac)
    names(cl) <- rownames(colData(cds))

    truth <- as.character(colData(cds)[[ct_key]])

    # Metrics
    ari <- mclust::adjustedRandIndex(cl, truth)
    nmi <- aricode::NMI(cl, truth)
    ami <- aricode::AMI(cl, truth)

    out_dir <- file.path(result_root, tissue)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

    df <- data.frame(
      cell = rownames(colData(cds)),
      cluster = cl[rownames(colData(cds))],
      label = truth,
      stringsAsFactors = FALSE
    )
    write.csv(df, file.path(out_dir, "clusters.csv"), row.names = FALSE)

    end_time <- Sys.time()
    running_time <- difftime(end_time, start_time, units = "secs")

    writeLines(
      c(
        paste0("Tissue: ", tissue),
        "Method: Monocle3",
        paste0("ARI: ", ari),
        paste0("NMI: ", nmi),
        paste0("AMI: ", ami),
        paste0("Max_cells: ", max_cells),
        paste0("Seed: ", seed),
        paste0("Counts_source: ", counts_source),
        paste0("Time: ", round(as.numeric(running_time), 2))
      ),
      file.path(out_dir, "results.txt")
    )

    message(sprintf("[OK] %s -> %s (%.2f sec)", tissue, file.path(out_dir, "results.txt"), as.numeric(running_time)))
    invisible(TRUE)

  }, error = function(e) {
    msg <- sprintf("Error processing tissue %s: %s", tissue, e$message)
    message(paste0("[ERR] ", msg))
    write(msg, file = file.path(result_root, "error_log.txt"), append = TRUE)
    invisible(FALSE)
  })
}

# -----------------------------
# Discover tissues + run
# -----------------------------
# tissues <- list.dirs(base_path, full.names = FALSE, recursive = FALSE)
tissue_file <- "/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/new_cellxgene_download/meta-data/valid_tissues_full_data.txt"

# Load tissues from file (one tissue per line)
tissues <- readLines(tissue_file, warn = FALSE)
tissues <- trimws(tissues)
tissues <- tissues[nzchar(tissues)]                 # drop empty lines
tissues <- tissues[!startsWith(tissues, "#")]       # optional: drop comments

if (!is.na(only_dataset)) tissues <- tissues[tissues == only_dataset]
remain <- tissues[!file.exists(file.path(result_root, tissues, "results.txt"))]

message(sprintf("Tissues to process: %d (parallel=%s n_procs=%d)", length(remain), parallel_on, n_procs))

if (parallel_on && length(remain) > 1) {
  n_use <- max(1L, min(n_procs, length(remain)))
  mclapply(remain, process_tissue, mc.cores = n_use)
} else {
  for (t in remain) process_tissue(t)
}

message("DONE")

# Example:
# Rscript run_monocle3.R \
#   --data_dir /data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells \
#   --result_dir /data/share/dungp/single-cell/clustering-results-with-running-time/monocle3 \
#   --max_cells 30000000 --seed 1 \
#   --parallel 1 --n_procs 4
