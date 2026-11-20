# CellFM runner

This directory retains the project-specific per-tissue fine-tuning runner
`run_CellFM_single_tissue.py`. The CellFM-torch source, pretrained weights,
input data, and generated results are intentionally excluded from this repo.

## Setup

Clone the upstream implementation and follow its environment and checkpoint
instructions:

```bash
git clone https://github.com/biomed-AI/CellFM-torch.git
cd CellFM-torch
```

Copy this runner into that checkout, or ensure the checkout is on `PYTHONPATH`.
The runner imports `layers.utils` and `model`, so it must execute against the
upstream CellFM-torch source tree. Download the CellFM 80M checkpoint and update
`cfg.ckpt_path` near the top of the runner to its local location.

## Inputs and execution

The configured data root must contain `<tissue>/train.h5ad` and
`<tissue>/test.h5ad`. Edit the `data_root` and `save_root` variables in the
runner for the local environment. A single-tissue GPU run is:

```bash
python run_CellFM_single_tissue.py --tissue tissue_name --device 0
```

Omit `--tissue` to process all valid tissue folders. `--parallel` and
`--max-workers` enable local multiprocessing, but single-tissue mode takes
priority and is the safer choice for external scheduling.

Each completed tissue receives `result.txt`, `timing.json`, prediction data, and
a model checkpoint under `<save_root>/<tissue>/`. Keep output outside this repo.
