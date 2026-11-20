# Intra-cell-type annotation

This directory contains the project-specific scripts used to train and evaluate
within-tissue cell-type classifiers. Each tissue is expected to have this layout:

```text
<data-root>/<tissue>/train.h5ad
<data-root>/<tissue>/test.h5ad
```

The scripts use cell-type labels from `cell_type_ontology_term_id` (with special
cases handled in the RF/LGBM scripts). Results are written below
`<save-root>/<tissue>/`; a `result.txt` file marks a completed tissue and lets
batch runners resume safely.

## Methods

| Directory | Retained runner(s) | Notes |
| --- | --- | --- |
| `RF_GBM` | `run_RF_GBM.py` and batch launchers | Classical baselines: random forest and LightGBM. |
| `cellFM` | `run_CellFM_single_tissue.py` | Fine-tunes CellFM per tissue; requires the external CellFM-torch checkout and 80M checkpoint. |
| `scFoundation` | `run_scFoundation_single_tissue.py` | Fine-tunes/evaluates scFoundation per tissue; requires the external source checkout and weights. |
| `scTab` | `run_scTab.py` | Runs the retained scTab benchmark implementation; requires the external scTab checkout. |

External source repositories, checkpoints, data, logs, and results are not
versioned here. See each method README for setup and execution details.

## RF and LightGBM

`run_RF_GBM.py` is the large-data runner. It supports
`--model rf` and `--model lgbm`, optional stratified training subsampling, and
per-tissue multiprocessing. For a single tissue:

```bash
python RF_GBM/run_RF_GBM.py \
  --data_root /path/to/tissues \
  --save_root /path/to/results/rf \
  --model rf --tissue tissue_name \
  --use_subsample --subsample_n_cells 500000
```

The shell launchers select tissues above a configurable cell-count threshold, enforce a
timeout, collect runtime logs, and resume unfinished tissues. Edit their
`DATA_ROOT`, `SAVE_ROOT`, `COUNT_CSV`, resource limits, and process count before
use. LightGBM must be installed for `--model lgbm`.

## Reproducibility

Use the same data root, tissue list, seed, and subsampling limit across methods
when comparing results. The runners record accuracy, macro F1, precision,
recall, and runtime in each tissue result directory. Keep outputs outside this
repository; the root `.gitignore` excludes common generated artifacts.
