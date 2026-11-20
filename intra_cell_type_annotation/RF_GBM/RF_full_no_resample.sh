#!/bin/bash
set -u

DATA_ROOT="/data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells"
SAVE_ROOT="/data/share/dungp/single-cell/ct-annot-results/RF_full_may5_over_300k"
LOG_ROOT="${SAVE_ROOT}/runtime_logs"

COUNT_CSV="/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/new_cellxgene_download/meta-data/tissue_cell_counts.csv"

N_PROCS=2
TIMEOUT_HOURS=200
MIN_CELLS=300000
MODEL="rf"

TIME_BIN="$(which time)"

mkdir -p "${LOG_ROOT}"

TISSUE_LIST="${LOG_ROOT}/tissues_gt_${MIN_CELLS}.txt"

# select tissues with test_cells > 300000
python - <<PY
import pandas as pd

df = pd.read_csv("${COUNT_CSV}")

cell_col = "train_cells"  # change to "train_cells" if needed

if "tissue" not in df.columns:
    raise ValueError(f"Cannot find tissue column. Columns: {df.columns.tolist()}")

if cell_col not in df.columns:
    raise ValueError(f"Cannot find {cell_col}. Columns: {df.columns.tolist()}")

df = df[df[cell_col] > ${MIN_CELLS}]
df["tissue"].dropna().astype(str).to_csv("${TISSUE_LIST}", index=False, header=False)

print(f"Selected {len(df)} tissues with {cell_col} > ${MIN_CELLS}")
PY

run_one_tissue () {
    TISSUE="$1"
    LOG_FILE="${LOG_ROOT}/${TISSUE}.log"

    echo "=================================================="
    echo "Running tissue: ${TISSUE}"
    echo "Log: ${LOG_FILE}"
    echo "=================================================="

    if [ -f "${SAVE_ROOT}/${TISSUE}/result.txt" ] || [ -f "${SAVE_ROOT}/${TISSUE}/results.txt" ]; then
        echo "[SKIP] ${TISSUE} already finished" | tee -a "${LOG_FILE}"
        return 0
    fi

    "$TIME_BIN" -v timeout "${TIMEOUT_HOURS}h" python run_RF_GBM.py \
        --data_root "${DATA_ROOT}" \
        --save_root "${SAVE_ROOT}" \
        --model "${MODEL}" \
        --use_subsample \
        --subsample_n_cells 600000 \
        --tissue "${TISSUE}" \
        > "${LOG_FILE}" 2>&1

    EXIT_CODE=$?

    if [ ${EXIT_CODE} -eq 124 ]; then
        echo "[TIMEOUT] ${TISSUE} exceeded ${TIMEOUT_HOURS} hours" | tee -a "${LOG_FILE}"
    elif [ ${EXIT_CODE} -ne 0 ]; then
        echo "[ERROR] ${TISSUE} failed with exit code ${EXIT_CODE}" | tee -a "${LOG_FILE}"
    else
        echo "[OK] Finished: ${TISSUE}" | tee -a "${LOG_FILE}"
    fi

    grep -E "Elapsed|Maximum resident|Exit status|Command terminated|Killed|MemoryError|cannot allocate" "${LOG_FILE}" || true
}

export -f run_one_tissue
export DATA_ROOT SAVE_ROOT LOG_ROOT TIME_BIN TIMEOUT_HOURS MODEL

tac "${TISSUE_LIST}" | xargs -I{} -P "${N_PROCS}" bash -c 'run_one_tissue "$@"' _ {}

echo "All selected RF tissues completed."