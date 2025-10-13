#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n scholarly -y python=3.11
conda activate scholarly
conda install -y pandas matplotlib numpy requests tqdm scikit-learn

########## CONFIG ##########
# List your inputs here:
INPUTS=(
  "input/wop.csv"
  "input/main_papers.csv"
)

# General settings
ENV_NAME="scholarly"
OUT_BASE="output"
YEAR_COLUMN="year"
SLEEP="1"
MAX_CITERS=1000
PROGRESS_EVERY="25"
LOG_LEVEL="INFO"
TOP_N="20"
START_YEAR="2005"
END_YEAR="2024"
SKIP_EXISTING=0   # 1 => pass --skip-existing to OpenAlex step
############################

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH." >&2
  exit 1
fi

run_one() {
  local INPUT="$1"
  if [[ ! -f "$INPUT" ]]; then
    echo "Input not found: $INPUT" >&2
    return 1
  fi

  local BASE SLUG
  BASE="$(basename "$INPUT")"
  SLUG="${BASE%.*}"
  # simple slugging: spaces -> underscores
  SLUG="${SLUG// /_}"

  local DATA_DIR="${OUT_BASE}/${SLUG}"
  local MAIN_OUT="${DATA_DIR}/with_citations.csv"
  local SUMMARY_OUT="${DATA_DIR}/reuse_summary.csv"
  local SUMMARY_OUT_CLASSIFIED="${DATA_DIR}/reuse_summary-classified.csv"
  local FIG_DIR="${DATA_DIR}/figures"
  local LOG_DIR="${DATA_DIR}/logs"
  local LOG_FILE="${LOG_DIR}/run_openalex.log"
  local PATTERN_LIBRARY_DIR="../MultiSource/patterns-repository"

  mkdir -p "$DATA_DIR" "$FIG_DIR" "$LOG_DIR"

  local SKIP_FLAG=()
  if [[ "$SKIP_EXISTING" -eq 1 ]]; then
    SKIP_FLAG=(--skip-existing)
  fi

  export PYTHONIOENCODING=utf-8

  echo "==> [${SLUG}] Step 1: paper_citations_openalex.py"
  conda run -n "$ENV_NAME" python -u paper_citations_openalex.py \
    --input "$INPUT" \
    --main-out "$MAIN_OUT" \
    --data-dir "$DATA_DIR" \
    --year-column "$YEAR_COLUMN" \
    --sleep "$SLEEP" \
    --max-citers "$MAX_CITERS" \
    --progress-every "$PROGRESS_EVERY" \
    --log-file "$LOG_FILE" \
    --log-level "$LOG_LEVEL" \
    "${SKIP_FLAG[@]}"

  echo "==> [${SLUG}] Step 2: classify_odp_reuse.py"
  conda run -n "$ENV_NAME" python -u classify_odp_reuse.py \
    --data-dir "$DATA_DIR" \
    --pattern-library-dir "$PATTERN_LIBRARY_DIR" \
    --log-dir "$LOG_DIR" \
    --out-summary "$SUMMARY_OUT" \
    --out-all-classified "$SUMMARY_OUT_CLASSIFIED" \
    --log "$LOG_LEVEL"

  echo "==> [${SLUG}] Step 3: odp_figures.py"
  conda run -n "$ENV_NAME" python -u odp_figures.py \
    --data-dir "$DATA_DIR" \
    --summary-file "$SUMMARY_OUT" \
    --fig-dir "$FIG_DIR" \
    --top-n "$TOP_N" \
    --start-year "$START_YEAR" \
    --end-year "$END_YEAR"

  echo "==> [${SLUG}] Step 4: odp_summary_and_plots.py"
  conda run -n "$ENV_NAME" python -u odp_summary_and_plots.py \
  --input "$INPUT" \
  --data-dir "$DATA_DIR" \
  --fig-dir "$FIG_DIR" \
  --top-n "$TOP_N" \
  --log-level "$LOG_LEVEL" \
  --start-year "$START_YEAR" \
  --end-year "$END_YEAR"
  
  echo "✔ Done: ${SLUG}"
}

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "No inputs configured in the script (INPUTS array is empty)." >&2
  exit 1
fi

for INPUT in "${INPUTS[@]}"; do
  echo "=== Processing ${INPUT} ==="
  run_one "$INPUT"
done

echo "All done."
