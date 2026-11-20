# scFoundation runner

This directory retains the project-specific runner
`run_scFoundation_single_tissue.py`; upstream source, model weights, input data,
and results are not versioned here.

## Setup

Obtain the implementation and follow its upstream installation/model-weight
instructions:

```bash
git clone https://github.com/biomap-research/scFoundation.git
cd scFoundation
```

Run the retained script from the configured scFoundation checkout. Its model path
defaults to `./model/models1.ckpt`, so download the matching pretrained weight
and either place it there or update the path in the runner. The runner also uses
scFoundation fine-tuning dataset/module helpers, which must be available from
the local environment. In particular, this runner imports `wrapper.SCFoundationLitModule` and
`dataset.H5ADDataset`, `LOG1PTransform`, and `TotalSumNormalize` at import time.
These integration helpers are not bundled in this directory; provide the original
fine-tuning helpers on `PYTHONPATH` before execution. `--workdir` alone does not
make those imports available.

## Inputs and execution

Input data must be arranged as `<data-root>/<tissue>/train.h5ad` and
`<data-root>/<tissue>/test.h5ad`. Select exactly one tissue source: `--tissue`,
`--tissues-file` (one tissue name per line), or `--all`.

```bash
python run_scFoundation_single_tissue.py \
  --workdir /path/to/scFoundation \
  --data-root /path/to/tissues \
  --save-root /path/to/results/scFoundation \
  --tissue tissue_name \
  --device 0 --batch-size 32 --max-epochs 30 --skip-finished
```

`--device` sets `CUDA_VISIBLE_DEVICES`; the runner then uses GPU index 0 within
that visible-device set. `--pool-type` accepts `mean`, `max`, or `all`.
Completed tissues contain `result.txt`, metrics, and prediction/embedding output
below `<save-root>/<tissue>/`; keep those outputs outside this repo.
