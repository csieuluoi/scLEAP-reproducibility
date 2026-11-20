#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=6

BASE_TOKENIZED="/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/zero_shot_other_models/LangCell/data/tokenized_dataset"
CKPT_DIR="./ckpt"
OUT_DIR="./results"
SCRIPT="run_langcell_zeroshot.py"

DATASETS=(
    "GSE130148"
    "GSE84133"
    "GSE185948"
    "E-CURD-114.project"
    "E-HCAD-9.project"
    "E-HCAD-10.project"
    "E-MTAB-9221.project"
    "GSE85241"
    "E-MTAB-5061"
    "GSE81608"
    "GSE86473"
    "E-MTAB-8410.project"
    "E-GEOD-134144.project"
    "E-MTAB-9543.project"
    "E-HCAD-15.project"
    "GSE149938"
)
# DATASETS=(
#   "GSE185948"
# )

# DATASETS=(
#   "GSE130148"
#   "GSE149938"
# )

mkdir -p "${OUT_DIR}"

for d in "${DATASETS[@]}"; do
  TOKENIZED_PATH="${BASE_TOKENIZED}/${d}"
  LABELS_PATH="${TOKENIZED_PATH}/candidate_labels.txt"

  if [[ ! -d "${TOKENIZED_PATH}" ]]; then
    echo "[SKIP] Missing tokenized dataset: ${TOKENIZED_PATH}"
    continue
  fi

  if [[ ! -f "${LABELS_PATH}" ]]; then
    echo "[SKIP] Missing candidate labels: ${LABELS_PATH}"
    continue
  fi

  echo "========================================"
  echo "Running ${d}"
  echo "========================================"

  python "${SCRIPT}" \
    --tokenized_path "${TOKENIZED_PATH}" \
    --candidate_labels "${LABELS_PATH}" \
    --ckpt_dir "${CKPT_DIR}" \
    --output_csv "${OUT_DIR}/${d}_preds.csv" \
    --output_pt "${OUT_DIR}/${d}_outputs.pt" \
    2>&1 | tee "${OUT_DIR}/${d}.log"

  echo "[DONE] ${d}"
done