# Environment setup

Use separate environments for scLEAP, R plotting, and baselines with conflicting dependencies. The environment files provide dependency specifications, not a frozen record of the original experiment environment.

## scLEAP

From the reproducibility repository root:

```bash
conda env create -f scLEAP/scleap.yml
conda activate scleap
```

Install PyTorch and GPU dependencies for your CUDA/driver configuration using the instructions in the [included package README](scLEAP/README.md). Then install the local source:

```bash
python -m pip install -e ./scLEAP
```

The environment specifies Python 3.11; package metadata pins Transformers to 4.55.3. PyTorch, RAPIDS, and FAISS require separate platform-specific setup. Model downloads may also be needed for the text encoder. Do not assume that installing the core package installs every benchmark dependency.

## Figures and statistical tables

For Python plotting/statistics, the main dependencies are NumPy, pandas, SciPy, scikit-learn, matplotlib, PyArrow, PyYAML, and statsmodels. AnnData/Scanpy are needed for scripts that read single-cell matrices. These are covered by the scLEAP environment or can be installed in a separate plotting environment.

For R figure and table scripts:

```r
install.packages(c(
  "ggplot2", "dplyr", "tidyr", "readr", "patchwork",
  "cowplot", "ggrepel", "scales", "stringr"
))
```

`grid` is part of R. Clustering runners additionally require method-specific R packages; see [clustering/README.md](clustering/README.md).

## Baselines

Follow the method-specific READMEs under [within-tissue annotation](intra_cell_type_annotation/README.md), [zero-shot annotation](zero_shot_annotation/README.md), and [clustering](clustering/README.md). Install each upstream implementation and obtain its matching model weights before using the project runner. Record the upstream commit and installed versions with any new run.

Batch launchers assume Bash, GNU `time`, and a Linux-like environment. Review GPU IDs, worker counts, memory/cell limits, and paths before launching full experiments. Training and large-scale clustering can require substantial GPU memory, RAM, and storage; choose resources according to the selected dataset and method.
