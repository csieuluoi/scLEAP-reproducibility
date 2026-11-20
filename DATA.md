# Data, checkpoints, and saved results

Download the experiment assets from **[the scLEAP Seafile archive](https://seafile.tinnguyen-lab.com/d/78a0f3c9084045768507/)**. Download only the files needed for your analysis; training matrices and checkpoints can be large. The shared page's remote file inventory has not been verified during this repository cleanup.

Keep downloaded files out of Git. For plotting, use this local layout relative to the reproducibility repository root:

```text
resources/
└── scLEAP/
    ├── data/
    │   ├── meta-data/
    │   └── zero_shot_datasets/
    ├── checkpoints/
    │   └── foundation_model/best.ckpt
    └── results/
        ├── tables/
        └── foundation_model_results/
outputs/
└── tables/                     # derived statistical summaries
figure/                         # generated figures
```

If the download already contains a top-level `scLEAP/` directory, put that directory inside `resources/` rather than nesting it twice. The layout above describes local destinations expected by the plotting scripts, not a verified inventory of the remote share.

## Assets by workflow

| Workflow | Required assets |
| --- | --- |
| Within-tissue annotation | Matched `<tissue>/train.h5ad` and `<tissue>/test.h5ad`; or the scLEAP Parquet layout below |
| Foundation-model training | `train_parquets/`, `train_ontology_to_int.json`, ontology descriptions, and the graph embedding NPZ when enabled |
| scLEAP zero-shot prediction | Released foundation-model checkpoint and external-dataset H5AD files |
| Baseline zero-shot prediction | Method-specific checkpoint; LangCell tokenized datasets and candidate labels, or CellWhisperer preprocessed H5AD artifacts |
| Clustering | `<tissue>/test.h5ad`, tissue list, and a checkpoint for scLEAP |
| Figures/statistics | Saved result CSVs; additional coordinates, predictions, or timing logs for the panels that use them |

For H5AD annotation inputs, runners generally use `obs['cell_type_ontology_term_id']`; clustering defaults to `obs['cell_type']`. Preserve the released cell identifiers, gene order, preprocessing, and label mappings. scTab and other foundation-model baselines can require model-specific gene alignment; their inputs are not interchangeable merely because filenames match.

For scLEAP Parquet input, each tissue contains:

```text
<tissue>/
├── train_parquets/
├── test_parquets/
├── train_ontology_to_int.json
├── train_celltype_to_int.json
├── all_ontology_to_int.json
└── all_celltype_to_int.json
```

Pass absolute paths to experiment entry points where possible. Consult the [analysis guide](scLEAP/analyses/README.md) for argument names; scLEAP's commands resolve relative paths against the included `scLEAP/` directory.

## Plotting inputs

The main saved tables referenced by figure/statistics scripts are:

- `cell_type_classification.csv`, `celltype_counts.csv`
- `donor_celltype_bins_ALL_METHODS.csv`, `donor_celltype_metrics_ALL_METHODS.csv`
- `Fig2_zero_shot_all_cells_truth_pred_dataset.csv`, `Fig2_zero_shot_all_metrics_summary.csv`
- `cell_type_clustering.csv`, `Fig3_topK_celltype_metrics.csv`

Place these under `resources/scLEAP/results/tables/`. Derived tables such as `per_tissue_full_metrics.csv`, `zeroshot_*_bootstrap.csv`, and runtime summaries belong in `outputs/tables/`; use released copies if available or generate them with the scripts described in the [plotting guide](plotting/README.md).

For the landscape panel, arrange Parquet label chunks under `all_results/1M_cells_70_cell_types/` and method coordinates under `all_results/landscape_tsne_umap/`. Update `plotting/config/cluster_configs_all_representations.json` if your coordinate directories differ. Other inputs, including metadata lists, zero-shot prediction files, and timing logs, are described in the corresponding scripts and plotting guide.

## Restoring package metadata

Only scripts, documentation, and configuration are tracked in Git. Tissue-count
CSVs and ontology-description JSON files are distributed separately as data.
Before running scLEAP, restore these files from the released assets into the
local source tree:

```text
scLEAP/src/scleap/ct_descriptions/cell_types_info.json
scLEAP/src/scleap/ct_descriptions/cell_name_to_ols_id.json
scLEAP/data/tissue_cell_counts.csv
```

The first two are package metadata used by the model; the tissue CSV supplies
the default clustering tissue list. You can also pass a tissue list explicitly
using `--tissue_csv`. Keep these local files out of version control.
