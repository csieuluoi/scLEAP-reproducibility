# Clustering benchmark

This directory contains runners for Scanpy, Pegasus, scVI+Leiden, Seurat,
Monocle3, and SHARP. All methods use the same tissue-level input layout:

```text
<data-dir>/<tissue>/test.h5ad
```

The ground-truth column is `cell_type`. Each method deterministically selects
at most `--max_cells` cells using the supplied seed and stores selected cell IDs
in `<result-dir>/_subsamples/`. Reusing that directory, seed, and cell limit
ensures all methods compare the same cells.

## Outputs

Each successful run writes `<result-dir>/<tissue>/results.txt`, which is also
the resume marker used by all batch launchers. It records NMI, ARI, AMI, runtime,
and method settings. Per-cell cluster assignments are saved as `cluster_df.csv`
or `clusters.csv` for the Python methods; R methods write method-specific output
alongside `results.txt`. Keep `result-dir` outside this repository.

## Methods and dependencies

| Method | Runner | Main dependencies |
| --- | --- | --- |
| Scanpy | `run_scanpy.py` / `run_scanpy.sh` | Python, Scanpy, AnnData, scikit-learn |
| Pegasus | `run_pegasus.py` / `run_pegasus.sh` | Python, `pegasuspy`, scikit-learn |
| scVI+Leiden | `run_scvi.py` / `run_scvi.sh` | Python, Scanpy, PyTorch, `scvi-tools` |
| Seurat | `run_seurat.R` / `run_seurat.sh` | R, Seurat, anndata, mclust, aricode |
| Monocle3 | `run_monocle3.R` / `run_monocle3.sh` | R, monocle3, anndata, mclust, aricode |
| SHARP | `run_sharp.R` / `run_sharp.sh` | R, SHARP, SingleCellExperiment, ClusterR, anndata |

The R runners also use `RhpcBLASctl`, Matrix, parallel, and digest; Seurat also
uses future. The scripts set BLAS/OpenMP threads to one to prevent resource
oversubscription during tissue-level parallelism.

## Run one tissue

Run from this directory. Example using Scanpy:

```bash
python run_scanpy.py \
  --data_dir /path/to/tissues \
  --result_dir /path/to/results/scanpy \
  --dataset tissue_name \
  --max_cells 3000000 --seed 1
```

The other Python and R runners use the same core options: `--data_dir`,
`--result_dir`, `--dataset`, `--max_cells`, and `--seed`. Consult `--help` or
the script header for method-specific options. Pegasus supports optional
UMAP/silhouette calculations. scVI additionally exposes epoch, highly-variable
gene, resolution, count-clipping, and accelerator settings.

## Batch launchers

Each `run_<method>.sh` launcher reads a tissue-list file, skips tissues with an
existing `results.txt`, executes the matching runner, and writes runtime logs.
Before use, edit `FULL_LIST`, `DATA_DIR`, `RESULT_DIR`, and resource settings at
the top of the chosen launcher. `run_scvi.sh` also sets GPU ID, epochs, number
of top genes, clustering resolution, and count clipping.

```bash
bash run_scanpy.sh
```

The launchers rely on GNU `time` and relative script paths, so invoke them from
`clustering/` or adjust the paths.
