#!/bin/bash
set -u

FULL_LIST="/data/dungp/projects/single-Cell/Sy-Code/code-for-reproducibility/new_cellxgene_download/meta-data/valid_tissues_full_data.txt"
DATA_DIR="/data/share/dungp/single-cell/data/cxg_data_1M/splited_by_tissue_all_data_no_filter_v3_grouped_shards_v2_all_cells"
RESULT_DIR="/data/share/dungp/single-cell/clustering-results-with-running-time/pegasus"
LOG_DIR="${RESULT_DIR}/_logs"

MAX_CELLS=30000000
SEED=1
LABEL_KEY="cell_type"

TIME_BIN="$(which time)"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

DONE="${RESULT_DIR}/done_tissues.txt"
MISSING="${RESULT_DIR}/missing_tissues.txt"
SUMMARY="${LOG_DIR}/summary_missing.log"

find "$RESULT_DIR" -mindepth 2 -maxdepth 2 -name "results.txt" \
  | awk -F'/' '{print $(NF-1)}' \
  | sort -u > "$DONE"

grep -v '^[[:space:]]*$' "$FULL_LIST" \
  | sort -u \
  | comm -23 - "$DONE" \
  > "$MISSING"

echo "Done tissues:    $(wc -l < "$DONE")"
echo "Missing tissues: $(wc -l < "$MISSING")"
cat "$MISSING"
echo ""

while read -r tissue; do
  [[ -z "$tissue" ]] && continue

  if [[ -f "${RESULT_DIR}/${tissue}/results.txt" ]]; then
    echo "Skipping finished tissue: $tissue"
    continue
  fi

  echo "===== Running $tissue ====="

  "$TIME_BIN" -v python run_pegasus.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --label_key "$LABEL_KEY" \
    --max_cells "$MAX_CELLS" \
    --seed "$SEED" \
    --dataset "$tissue" \
    > "${LOG_DIR}/${tissue}.stdout.log" \
    2> "${LOG_DIR}/${tissue}.stderr_time.log"

  exit_code=$?

  {
    echo "===== $tissue ====="
    echo "$tissue exit code: $exit_code"
    grep -E "Maximum resident|Elapsed|Exit status|Command terminated|Error processing|Killed|cannot allocate|memory|MemoryError" \
      "${LOG_DIR}/${tissue}.stderr_time.log" || true
    echo ""
  } | tee -a "$SUMMARY"

done < "$MISSING"

echo "DONE"
echo "Summary log: $SUMMARY"