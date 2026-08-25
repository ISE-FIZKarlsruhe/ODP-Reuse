#!/usr/bin/env bash
# run_extraction.sh -- run every ODP extraction method over a corpus, then
# evaluate the ones that can be evaluated.
#
# Ground truth is optional. If an ontology directory carries a patterns/
# subdirectory whose entities overlap the ontology, that overlap is the answer
# key and the method is scored against it. If it does not -- as for a corpus of
# ontologies nobody has annotated -- the method still runs and its predicted
# groups are written out; only the scoring is skipped. The same command works
# either way, so a new corpus needs no new invocation.
#
# USAGE
#   ./run_extraction.sh --corpus DIR --out DIR [options]          # any corpus
#   ./run_extraction.sh --manifest M --data-root D --out DIR ...  # ODP-Reuse
#
# OPTIONS
#   --corpus DIR        directory of ontologies. Either one subdirectory per
#                       ontology (<name>/<name>.owl, optionally with
#                       <name>/patterns/), or a flat directory of RDF files.
#   --manifest FILE     ODP-Reuse MANIFEST.csv (with --data-root) instead
#   --data-root DIR     root of the ODP-Reuse corpus tree
#   --datasets "A B"    manifest mode only (default "github noOBO")
#   --out DIR           where results go (required)
#   --methods "A B"     default: every method plus the reference points
#   --with-full-cc      also run the complete conceptual-components pipeline
#                       (steps 1-7). Needs the Framester/UKB server -- see
#                       tools/run_asprino_full.py for how to stand it up.
#   --cc-server URL     step-3 server (default http://localhost:8080)
#   --cc-top-level F    topLevelAnchestors-prefix.csv from the upstream clone
#   --cc-k N            clusters for the cross-corpus step (0 = elbow)
#   --workers N         disambiguation concurrency (default 8)
#   --asprino-max-entities N
#                       skip conceptual-component extraction above this size
#                       (0 = no limit, the default for the ODP-Reuse corpus).
#                       Its cost grows steeply with the intensional graph, so a
#                       corpus containing very large ontologies needs a bound.
#                       Every skip is recorded, never silent.
#   --python BIN        python to use (default python)
#
# OUTPUT (under --out)
#   methods_results.csv        aggregate scores, where ground truth existed
#   methods_per_ontology.jsonl one row per (ontology, method) scored
#   predictions.jsonl          every predicted cluster, scored or not
#   methods_skipped.csv        every exclusion, with its reason
#   asprino-full/              the conceptual-components pipeline, if requested
#   SUMMARY.txt               what ran, what was scored, what was not

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python}"
CORPUS=""; MANIFEST=""; DATA_ROOT=""; OUT=""
DATASETS="github noOBO"
METHODS="all_in_one singletons connected_components namespace asprino_cc law_ap law_cfp"
WITH_FULL_CC=0
CC_SERVER="http://localhost:8080"
CC_TOP_LEVEL=""
CC_K=0
WORKERS=8
ASPRINO_MAX_ENT=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --corpus)       CORPUS="$2"; shift 2 ;;
    --manifest)     MANIFEST="$2"; shift 2 ;;
    --data-root)    DATA_ROOT="$2"; shift 2 ;;
    --datasets)     DATASETS="$2"; shift 2 ;;
    --out)          OUT="$2"; shift 2 ;;
    --methods)      METHODS="$2"; shift 2 ;;
    --with-full-cc) WITH_FULL_CC=1; shift ;;
    --cc-server)    CC_SERVER="$2"; shift 2 ;;
    --cc-top-level) CC_TOP_LEVEL="$2"; shift 2 ;;
    --cc-k)         CC_K="$2"; shift 2 ;;
    --workers)      WORKERS="$2"; shift 2 ;;
    --asprino-max-entities) ASPRINO_MAX_ENT="$2"; shift 2 ;;
    --python)       PY="$2"; shift 2 ;;
    -h|--help)      sed -n '2,45p' "$0"; exit 0 ;;
    *)              EXTRA+=("$1"); shift ;;
  esac
done

[[ -z "$OUT" ]] && { echo "error: --out is required" >&2; exit 2; }
if [[ -z "$CORPUS" && ( -z "$MANIFEST" || -z "$DATA_ROOT" ) ]]; then
  echo "error: give --corpus DIR, or --manifest and --data-root" >&2; exit 2
fi
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"

log() { echo "$*" | tee -a "$SUMMARY"; }
log "run_extraction.sh  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
log "methods: $METHODS"

# ---------------------------------------------------------------- source args
SRC=()
if [[ -n "$CORPUS" ]]; then
  SRC=(--corpus "$CORPUS")
  log "corpus: $CORPUS  (ground truth used where patterns/ provides it)"
else
  # shellcheck disable=SC2206
  DS_ARR=($DATASETS)
  SRC=(--manifest "$MANIFEST" --data-root "$DATA_ROOT" --datasets "${DS_ARR[@]}")
  log "manifest: $MANIFEST   data-root: $DATA_ROOT   datasets: $DATASETS"
fi

# ------------------------------------------------- the methods + the baselines
# shellcheck disable=SC2206
M_ARR=($METHODS)
log ""
log "== extraction methods and reference points =="
"$PY" -u "$HERE/tools/run_method_eval.py" \
    "${SRC[@]}" --out "$OUT" --methods "${M_ARR[@]}" \
    --emit-predictions --asprino-max-entities "$ASPRINO_MAX_ENT"     "${EXTRA[@]+"${EXTRA[@]}"}" 2>&1 | tee -a "$SUMMARY"
RC=${PIPESTATUS[0]}
[[ $RC -ne 0 ]] && log "[warn] run_method_eval.py exited $RC"

# ---------------------------------------------- conceptual components, 1..7
if [[ $WITH_FULL_CC -eq 1 ]]; then
  log ""
  log "== conceptual components, complete pipeline =="
  if [[ -z "$CC_TOP_LEVEL" ]]; then
    log "[skip] --with-full-cc needs --cc-top-level <topLevelAnchestors-prefix.csv>"
  elif ! curl -s -o /dev/null -m 10 "$CC_SERVER/cd?text=test&senseInventory=fn17"; then
    log "[skip] no step-3 server at $CC_SERVER."
    log "       See tools/run_asprino_full.py for how to stand one up. If it"
    log "       runs where this script cannot reach it, run the three stages by"
    log "       hand: --stage documents, tools/framester_client.py, --stage score."
  else
    "$PY" -u "$HERE/tools/run_asprino_full.py" \
        "${SRC[@]}" --out "$OUT/asprino-full" \
        --server "$CC_SERVER" --top-level "$CC_TOP_LEVEL" \
        --k "$CC_K" --workers "$WORKERS" 2>&1 | tee -a "$SUMMARY"
  fi
fi

# ------------------------------------------------------------------- summarise
log ""
log "== outcome =="
if [[ -s "$OUT/methods_results.csv" ]]; then
  log "scored -> $OUT/methods_results.csv"
  column -s, -t < "$OUT/methods_results.csv" 2>/dev/null | tee -a "$SUMMARY" \
    || cat "$OUT/methods_results.csv" | tee -a "$SUMMARY"
else
  log "nothing was scored: no ontology in this corpus had ground truth."
fi
if [[ -s "$OUT/predictions.jsonl" ]]; then
  log "predictions -> $OUT/predictions.jsonl ($(wc -l < "$OUT/predictions.jsonl") rows)"
fi
if [[ -s "$OUT/methods_skipped.csv" ]]; then
  log "exclusions  -> $OUT/methods_skipped.csv ($(( $(wc -l < "$OUT/methods_skipped.csv") - 1 )) rows)"
fi
log "done."
