#!/usr/bin/env bash
# Resume Gemini adjudication until every item is judged.
#
# Gemini rate-limits with HTTP 503 under load, so a single pass rarely finishes
# all 331 items. Each pass appends to machine_verdicts_gemini.csv and --resume
# skips what is already judged, so repeated passes converge. Nothing is lost
# between passes and nothing is re-judged.
#
#   bash pipeline/resume_gemini.sh            # keep going until done
#   bash pipeline/resume_gemini.sh 3          # at most 3 passes
#
# Safe to run repeatedly and safe to interrupt.

set -u
cd "$(dirname "$0")" || exit 1

OUT=out/adjudication
FILE=$OUT/machine_verdicts_gemini.csv
TOTAL=$(($(wc -l < "$OUT/items.csv") - 1))
MAX_PASSES=${1:-40}
BACKOFF=30

for pass in $(seq 1 "$MAX_PASSES"); do
  have=$(( $(wc -l < "$FILE" 2>/dev/null || echo 1) - 1 ))
  if [ "$have" -ge "$TOTAL" ]; then
    echo "complete: $have/$TOTAL judged"
    exit 0
  fi
  echo "pass $pass: $have/$TOTAL judged, resuming ..."
  python -u adjudicate_labels.py judge --out "$OUT" --providers gemini --resume \
    >> out/adjudicate_gemini_resume.log 2>&1

  after=$(( $(wc -l < "$FILE" 2>/dev/null || echo 1) - 1 ))
  if [ "$after" -le "$have" ]; then
    # no progress at all: the API is refusing everything, so wait longer before
    # trying again rather than hammering it
    echo "  no progress; backing off ${BACKOFF}s"
    sleep "$BACKOFF"
    BACKOFF=$(( BACKOFF * 2 ))
    [ "$BACKOFF" -gt 600 ] && BACKOFF=600
  else
    echo "  +$(( after - have )) judged"
    BACKOFF=30
  fi
done

have=$(( $(wc -l < "$FILE" 2>/dev/null || echo 1) - 1 ))
echo "stopped after $MAX_PASSES passes: $have/$TOTAL judged"
echo "re-run this script to continue; two raters (GPT + Claude) are already complete."
