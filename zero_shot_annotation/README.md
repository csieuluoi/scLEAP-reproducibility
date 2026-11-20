# Zero-shot annotation runners

This directory contains only the project-specific runners used for the zero-shot
annotation benchmark.  It intentionally does not include upstream source code,
Python environments, model checkpoints, input data, or generated predictions.
Obtain each method from its original repository before running its retained
script.

## CellWhisperer

The runner is `cellwhisperer/run_zeroshot_annotation.py`.

1. Clone the upstream implementation with its required submodules, then install
   its environment:

   ```bash
   git clone --recurse-submodules https://github.com/epigen/cellwhisperer.git
   cd cellwhisperer
   bash envs/setup_pixi.sh
   ```

   The upstream README also documents a Conda-based setup if Pixi is not used.

2. Run the **upstream CellWhisperer preprocessing workflow** before using this
   project runner. The runner does not create its input H5AD artifacts. Start
   with raw, unnormalised counts at
   `resources/<dataset>/read_count_table.h5ad` in the upstream checkout. From
   the cloned checkout, select one or more dataset names and run the pipeline:

   ```bash
   cd src/cellxgene_preprocessing
   export CUDA_VISIBLE_DEVICES=0
   DATASETS='["<dataset_name>"]'
   pixi run snakemake --cores 12 --rerun-incomplete --config "datasets=$DATASETS"
   ```

   For multiple datasets, use a JSON-like list, for example
   `DATASETS='["GSE130148", "GSE149938"]'`. Keep only one `DATASETS=...`
   assignment active. With the Conda setup, use `snakemake --use-conda` in
   place of `pixi run snakemake` and retain the other arguments.

   This preprocessing step produces the benchmark inputs under
   `results/<dataset>/cellwhisperer_clip_v1/`, including
   `cellxgene.h5ad` and `leiden_umap_embeddings.h5ad`.

3. Copy `run_zeroshot_annotation.py` into
   `<checkout>/src/cellxgene_preprocessing/`, then run it from that directory
   so its `cellwhisperer` imports resolve to the upstream package:

   ```bash
   pixi run python run_zeroshot_annotation.py
   ```
4. Before execution, edit the script's `modelpath`, `dataset_names`, and input
   paths.  It expects, for each selected dataset, a
   `leiden_umap_embeddings.h5ad` file and a `cellxgene.h5ad` reference file.
   Predictions are written as `similarity_scores_per_cell.csv` alongside the
   dataset inputs.

The runner currently uses fixed paths rather than command-line options; update
them for the local checkout and data location.

## LangCell

The retained scripts are:

- `LangCell/run_langcell_zeroshot.py` — zero-shot inference with optional
  label harmonization (`--apply_label_mapping`).
- `LangCell/run_zeroshot.sh` — batch launcher for a configured dataset list.

1. Clone and install the upstream implementation:

   ```bash
   git clone https://github.com/PharMolix/LangCell.git
   cd LangCell
   pip install -r requirements.txt
   ```

2. Copy the retained scripts into that checkout.  The Python runners import
   `data_preprocess.utils`, which is supplied by the upstream repository.
3. Download/provide the LangCell checkpoint directory.  It must contain
   `cell_bert/`, `text_bert/`, `cell_proj.bin`, `text_proj.bin`, and
   `ctm_head.bin`.
4. Provide a tokenized Hugging Face dataset directory and a
   `candidate_labels.txt` file.  A single run is, for example:

   ```bash
   python run_langcell_zeroshot.py \
     --tokenized_path /path/to/tokenized/DATASET \
     --candidate_labels /path/to/tokenized/DATASET/candidate_labels.txt \
     --ckpt_dir /path/to/ckpt \
     --output_csv results/DATASET_preds.csv \
     --output_pt results/DATASET_outputs.pt \
     --apply_label_mapping
   ```

   Omit `--apply_label_mapping` for raw label
   scoring.  The batch launcher has editable `BASE_TOKENIZED`, `CKPT_DIR`,
   `OUT_DIR`, GPU, and dataset-list settings at its top.

All checkpoints, tokenized datasets, logs, CSV predictions, and tensor outputs
are excluded from version control by the local `.gitignore`.
