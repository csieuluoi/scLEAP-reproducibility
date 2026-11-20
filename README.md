# scLEAP reproducibility

Code for reproducing the annotation, zero-shot transfer, clustering, figures, and statistical analyses for **scLEAP: single-cell annotation using Expression–Language Alignment and Poincaré geometry**.

**[Download data, checkpoints, and result files](https://seafile.tinnguyen-lab.com/d/78a0f3c9084045768507/)** · [Data guide](DATA.md) · [Setup](SETUP.md) · [Figures and tables](plotting/README.md)

This repository contains the scLEAP source and analysis entry points, project-specific baseline runners, and manuscript plotting scripts. Datasets, ontology-description JSON files, model weights, and generated outputs are distributed separately through the download link above. Git contains scripts, documentation, and configuration only.

## Repository guide

| Directory | Contents |
| --- | --- |
| [`scLEAP/`](scLEAP/README.md) | Python package and scLEAP training/inference workflows (ontology descriptions downloaded separately) |
| [`intra_cell_type_annotation/`](intra_cell_type_annotation/README.md) | Within-tissue annotation baselines: random forest, LightGBM, scTab, CellFM, scFoundation |
| [`zero_shot_annotation/`](zero_shot_annotation/README.md) | LangCell and CellWhisperer zero-shot annotation runners |
| [`clustering/`](clustering/README.md) | Scanpy, Pegasus, scVI, Seurat, Monocle3, and SHARP clustering benchmarks |
| [`plotting/`](plotting/README.md) | Manuscript figure panels, statistical analyses, and supplementary tables |

The included scLEAP source is a snapshot of the package in this workspace. The main method repository is [tinnlab/scLEAP](https://github.com/tinnlab/scLEAP).

## Getting started

1. Follow [SETUP.md](SETUP.md) to prepare the environment for the selected method. Baselines have separate upstream dependencies.
2. Download the required assets from the shared archive and arrange them as described in [DATA.md](DATA.md).
3. To regenerate figures from saved results, start with [plotting/README.md](plotting/README.md). Training is unnecessary for this route.
4. To rerun experiments, use the relevant workflow guide below and supply local data/checkpoint/output paths.

| Task | scLEAP entry point |
| --- | --- |
| Within-tissue annotation | `scLEAP/analyses/intra_tissue_annotation/run_scLEAP.py` |
| Foundation-model training | `scLEAP/analyses/zero_shot_annotation/train_foundation_model.py` |
| Zero-shot prediction from a checkpoint | `scLEAP/analyses/zero_shot_annotation/run_zero_shot_prediction.py` |
| Cell clustering | `scLEAP/analyses/cell_clustering/run_clustering_scLEAP_new.py` |

See the [scLEAP analysis guide](scLEAP/analyses/README.md) for commands and input schemas. Run those commands from `scLEAP/`; run the figure/table commands from this repository's root. Baseline guides specify their own working directories.

## Reproduction notes

Use the released train/test splits, gene ordering, label mappings, checkpoints, and recorded settings when comparing methods. A script's defaults are starting points; they do not establish the settings of every manuscript experiment. Several baseline scripts retain editable path/configuration blocks from the original experiments; update those before execution. GPU software and numerical differences can affect results.

Some result files and internal method keys use historical names `scLEAD`, `scLMCT`, or `scLMCT_tsne30` for scLEAP. Keep those keys when reading existing results; renaming them can break joins and plot ordering.

This repository was prepared through source and documentation review. Analysis scripts were not executed as part of this cleanup, and numerical reproduction is not claimed by that review.

## Attribution

Please cite the accompanying scLEAP paper when using these workflows, and cite the original methods and datasets used in each analysis. The included scLEAP package retains its [Apache-2.0 license](scLEAP/LICENSE). External implementations, pretrained weights, and datasets retain their own licenses and usage terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
