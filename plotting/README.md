# Figures, statistical analyses, and tables

Run these scripts from the **reproducibility repository root**, with the inputs arranged as in [DATA.md](../DATA.md). Plotting reads downloaded tables under `resources/scLEAP/results/tables/`, reads/writes derived tables under `outputs/tables/`, and writes plots under `figure/`. Existing outputs can be overwritten.

Historical method identifiers such as `scLEAD` are retained in the result tables and statistical scripts. They refer to scLEAP; preserve these identifiers when loading the released CSVs.

## Figure scripts

Panel names below follow the script filenames. Use the manuscript's final captions when assembling panels.

| Script | Content and principal inputs |
| --- | --- |
| `fig1_panel_b_stacked_bar.R` | Cell-type composition; `celltype_counts.csv` and `resources/scLEAP/data/meta-data/valid_tissues_full_data.txt` |
| `fig1_panel_c_violin.R` | Within-tissue annotation distributions; `cell_type_classification.csv` |
| `fig1_panel_d_bars.R` | Annotation performance by cell abundance; donor/cell-type result tables |
| `fig2_panel_b_zeroshot_metrics.R` | Zero-shot dataset/group metrics with bootstrap intervals |
| `fig2_panel_b_samplesize.R` | Zero-shot sample-size strata with bootstrap intervals |
| `fig2_panel_c_ecdf.R` | Per-cell-type zero-shot F1 empirical distributions |
| `fig2_panel_d_pred_count_bias.R` | Prediction-count bias; zero-shot prediction/results files |
| `fig2_panels_cd_combined.R` | Combined zero-shot visualizations; dataset-level files under `resources/scLEAP/results/foundation_model_results/` |
| `fig3_panel_a_tsne.py` | Cell landscape; coordinate configuration and Parquet cell labels under `all_results/` |
| `fig3_panel_b_embedding_metrics.R` | Embedding-quality metrics; `outputs/tables/Fig3_topK_celltype_metrics.csv` |
| `fig3_panel_c_clustering.R` | Clustering metrics; `cell_type_clustering.csv` |
| `supp_fig_s3.R` | Supplementary zero-shot analyses using derived bootstrap tables |
| `supp_accuracy_f1_gap.R` | Accuracy/F1 comparison; `outputs/tables/per_tissue_full_metrics.csv` |
| `supp_runtime_classification_line.R` | Classification runtime; derived runtime and per-tissue metrics |

For example, after installing the R dependencies and placing the saved classification table:

```bash
Rscript plotting/fig1_panel_c_violin.R
```

For panel-specific inputs and output filenames, inspect the path/configuration block in the selected script. Dataset-level prediction figures need more than the aggregate CSV tables.

## Statistical summaries

Use released derived summaries in `outputs/tables/` if available. To regenerate the within-tissue and clustering summaries from saved primary tables:

```bash
python plotting/tables/within_tissue_statistical_tests.py
python plotting/tables/clustering_statistical_tests.py
```

`tables/zero_shot_statistical_analysis.py` computes zero-shot summaries from the saved truth/prediction table. For the bootstrap-based figure panels, use this dependency order:

1. `tables/ingest_sclead_predictions.py` creates the canonical scLEAP predictions.
2. `tables/ingest_langcell_predictions.py` and `tables/ingest_cellwhisperer_predictions.py` ingest the baseline predictions and align labels.
3. `tables/compute_zeroshot_bootstrap.py` creates per-dataset, group, sample-size, and per-cell-type bootstrap summaries (1,000 resamples, seed 1).
4. `tables/zero_shot_pairwise_tests.py` performs the paired zero-shot comparisons.
5. Run the zero-shot figure scripts or `tables/gen_supp_tables.py` as needed.

Baseline ingestion expects predictions under `resources/zero_shot_results/langcell_results/` and `resources/zero_shot_results/cellwhisperer_results/`. Review the exact filenames and alignment assumptions in the ingestion scripts. Preserve the released row ordering and cell identities; arbitrary prediction files are not necessarily compatible.

`tables/compute_embedding_metrics_scanpy_seurat.py` computes additional landscape metrics using coordinates and cell labels under `all_results/1M_cells_70_cell_types/`. For plotting from released embedding metrics, place the appropriate saved `Fig3_topK_celltype_metrics.csv` directly in `outputs/tables/` instead.

## Dataset and runtime tables

- `dataset_inventory_tables.R` and `tables/dataset_inventory_tables.R` format dataset inventory summaries; their derived inputs include `zeroshot_dataset_summary_v2.csv` and `per_tissue_full_metrics.csv` in `outputs/tables/`.
- `tables/runtime_cost_extract.py` reads per-method timing artifacts from `resources/annotation_results/` and `resources/clustering_results/` and creates runtime CSVs.
- `runtime_cost_table.R` and `tables/runtime_cost_table.R` format those runtime CSVs into LaTeX tables.
- `tables/gen_supp_tables.py` formats zero-shot bootstrap summaries and embedding metrics into supplementary LaTeX tables.

The root and `tables/` variants of the dataset/runtime table scripts are retained as separate formatting variants. Select the one matching the intended table. These are analysis scripts rather than a single automated pipeline; execute only the steps needed for the selected panel or table.
