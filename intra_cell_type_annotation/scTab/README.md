# scTab runner

This directory retains only the project-specific benchmark runner: `run_scTab.py`.
The upstream source, training environment, data, and generated checkpoints/results
are deliberately excluded.

## Setup

Clone and install the upstream implementation, including its GPU dependencies
when using CUDA:

```bash
git clone https://github.com/theislab/scTab.git
cd scTab
pip install -e .
```

Copy `run_scTab.py` into that checkout, or run it with the upstream project on
`PYTHONPATH`. It imports the upstream TabNet, dataset, and utility modules.

## Inputs and execution

Before running, edit the hard-coded `save_root`, tissue-count CSV, valid-tissue
list, data root, and ontology-mapping JSON paths near the bottom of the script.
The expected data layout is `<data-root>/<tissue>/train.h5ad` and
`<data-root>/<tissue>/test.h5ad`.

```bash
python run_scTab.py --gpu 0 --seed 1 --workers 4 --parallel
```

Small tissues run in a multiprocessing pool when `--parallel` is supplied;
large tissues run sequentially. `--cal_silhouette` and `--cal_umap` enable
optional evaluation/visualization calculations. Each completed tissue writes
`result.txt` under the configured save root; model checkpoints and prediction
files are outputs and should remain outside this repo.
