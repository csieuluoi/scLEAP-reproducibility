# External methods and attribution

The baseline directories contain project-specific experiment runners. Install the original implementations separately and follow their licenses, model-weight terms, and citation instructions.

| Method | Upstream project |
| --- | --- |
| scLEAP | https://github.com/tinnlab/scLEAP |
| scTab | https://github.com/theislab/scTab |
| CellFM | https://github.com/biomed-AI/CellFM-torch |
| scFoundation | https://github.com/biomap-research/scFoundation |
| LangCell | https://github.com/PharMolix/LangCell |
| CellWhisperer | https://github.com/epigen/cellwhisperer |

Clustering and classical baselines use Scanpy, Pegasus, scvi-tools, Seurat, Monocle3, SHARP, scikit-learn, and LightGBM. Their dependencies are listed in the workflow READMEs and script imports.

The bundled `scLEAP/` source retains its original Apache-2.0 license. That license does not grant rights to separately distributed datasets, checkpoints, or external software. Dataset accessions and source metadata should accompany use of the shared data. Historical method names in released result tables are preserved for compatibility.
