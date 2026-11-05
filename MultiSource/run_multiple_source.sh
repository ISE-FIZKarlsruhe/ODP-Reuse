#!/usr/bin/env bash
set -euo pipefail
# -----------------------------
# Load .env if present and export all variables
# -----------------------------
if [ -f ".env" ]; then
  set -a
  . ./.env
  set +a
fi

# -----------------------------
# Config
# -----------------------------
REPO_URL="https://github.com/odpa/patterns-repository.git"
REPO_DIR="patterns-repository"
PY_SCRIPT="odp_multi_source_finder.py"   # adjust if located elsewhere
OUT_CSV="report/results_github.csv"
PATTERNS_TXT="report/retrieved_patterns.txt"
PATTERNS_CSV="report/retrieved_patterns.csv"
SOURCES="github" #google,github,bioportal,matportal,lov

# Outputs for extraction & diagnostics
EXTRACT_REPORT="report/iri_extraction_report.csv"
DIRS_SUMMARY="report/dirs_summary.csv"
DIRS_NO_IRI="report/dirs_without_iri.txt"
FILES_NO_IRI="report/files_without_iri.txt"

# API pacing
GITHUB_MAX_PAGES="${GITHUB_MAX_PAGES:-5}"
GOOGLE_PAGES="${GOOGLE_PAGES:-1}"

# Keep only IRIs containing this substring (override with IRI_HOST_FILTER="")
IRI_HOST_FILTER="${IRI_HOST_FILTER:-}"
#IRI_HOST_FILTER="${IRI_HOST_FILTER:-ontologydesignpatterns.org}"

EXTRA_ARGS="${*:-}"  # forward any extra args

# -----------------------------
# Clone or update repository
# -----------------------------
if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[i] Updating existing $REPO_DIR ..."
  git -C "$REPO_DIR" pull --ff-only
else
  echo "[i] Cloning $REPO_URL ..."
  git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi

# -----------------------------
# Gather candidate files
# -----------------------------
echo "[i] Scanning for .owl/.ttl files ..."
mapfile -t FILES < <(find "$REPO_DIR" -type f \( -iname '*.owl' -o -iname '*.ttl' \) | sort)
TOTAL_FILES=${#FILES[@]}
if [[ $TOTAL_FILES -eq 0 ]]; then
  echo "[!] No .owl/.ttl files found under $REPO_DIR. Exiting."
  exit 1
fi
echo "[i] Found $TOTAL_FILES ontology files."

# -----------------------------
# Comprehensive IRI extraction (pattern_iris.py; the CI harvest uses the same module)
# -----------------------------
TMP_IRIS="$(mktemp)"
trap 'rm -f "$TMP_IRIS"' EXIT

python3 pattern_iris.py extract "$IRI_HOST_FILTER" "$EXTRACT_REPORT" "$DIRS_SUMMARY" \
  "$DIRS_NO_IRI" "$FILES_NO_IRI" "${FILES[@]}" > "$TMP_IRIS"

# Filter out generic template/homepage IRIs (optional)
grep -viE '/template(/.*)?$' "$TMP_IRIS" | \
grep -viE '^https?://(www\.)?ontologydesignpatterns\.org/?$' > "${TMP_IRIS}.filtered" || true
mv "${TMP_IRIS}.filtered" "$TMP_IRIS"

# -----------------------------
# Save retrieved IRIs
# -----------------------------
cp "$TMP_IRIS" "$PATTERNS_TXT"
{ echo "pattern_iri"; cat "$TMP_IRIS"; } > "$PATTERNS_CSV"

COUNT=$(wc -l < "$TMP_IRIS")
DIRS_WITH_NO_IRI=$(wc -l < "$DIRS_NO_IRI" || echo 0)
FILES_WITH_NO_IRI=$(wc -l < "$FILES_NO_IRI" || echo 0)

echo
echo "=============================================="
echo "[i] Extraction summary"
echo "    Total ontology files:       $TOTAL_FILES"
echo "    Unique IRIs extracted:      $COUNT"
echo "    Dirs with NO IRI matches:   $DIRS_WITH_NO_IRI  (see $DIRS_NO_IRI)"
echo "    Files with NO IRI matches:  $FILES_WITH_NO_IRI (see $FILES_NO_IRI)"
echo
echo "[i] Method frequencies (top 10):"
awk -F, 'NR>1 {cnt[$4]++} END {for (m in cnt) printf "%d,%s\n", cnt[m], m}' "$EXTRACT_REPORT" \
  | sort -t, -k1,1nr | head -n 10 | awk -F, '{printf "    %7d  %s\n", $1, $2}'
echo
echo "[i] Top directories by missing IRIs (top 15):"
awk -F, 'NR>1 {print $1","$4","$2}' "$DIRS_SUMMARY" | sort -t, -k2,2nr | head -n 15 \
  | awk -F, '{printf "    %3d missing of %3d  %s\n", $2, $3, $1}'
echo
echo "[i] First 20 IRIs (preview):"
head -n 20 "$PATTERNS_TXT" | sed 's/^/    /'
if [[ "$COUNT" -gt 20 ]]; then
  echo "    ... (see $PATTERNS_TXT for full list)"
fi
echo
echo "Saved files:"
echo "  - IRIs (txt):               $PATTERNS_TXT"
echo "  - IRIs (csv):               $PATTERNS_CSV"
echo "  - Per-file report:          $EXTRACT_REPORT"
echo "  - Per-directory summary:    $DIRS_SUMMARY"
echo "  - Dirs with no IRI:         $DIRS_NO_IRI"
echo "  - Files with no IRI:        $FILES_NO_IRI"
echo "=============================================="
echo

# -----------------------------
# Run the Python script
# -----------------------------
#"ontologydesignpatterns.org" (filetype:owl OR filetype:ttl OR filetype:rdf OR filetype:nt OR filetype:trig OR filetype:xml)
python3 "$PY_SCRIPT" \
  --patterns "ontologydesignpatterns" \
  --sources "google" \
  --github-per-page 50 \
  --github-max-pages "$GITHUB_MAX_PAGES" \
  --github-delay-ms 1400 \
  --github-max-retries 6 \
  --google-pages "$GOOGLE_PAGES" \
  --google-max-queries 90 \
  --google-skip-on-429 \
  --google-min-hits 2 \
  --skip-generic \
  --out odp_google.csv \
  $EXTRA_ARGS

# run: all major sources (auto-discovers ODP IRIs)
python3 "$PY_SCRIPT" \
  --patterns "$(paste -sd, "$PATTERNS_TXT")" \
  --sources "$SOURCES" \
  --github-per-page 50 \
  --github-max-pages "$GITHUB_MAX_PAGES" \
  --github-delay-ms 1400 \
  --github-max-retries 6 \
  --google-pages "$GOOGLE_PAGES" \
  --skip-generic \
  --out "$OUT_CSV" \
  $EXTRA_ARGS
  

# run: add SPARQL endpoint (e.g., LOV; replace with your endpoint)
python3 "$PY_SCRIPT" \
  --patterns "$(paste -sd, "$PATTERNS_TXT")" \
  --sources sparql \
  --sparql-endpoint https://lov.linkeddata.es/dataset/lov/sparql \
  --google-pages "$GOOGLE_PAGES" \
  --skip-generic \
  --out report/sparql_results.csv \
  $EXTRA_ARGS

echo "Downloading datasets and building gold standards ..."
python3 build_github_dataset.py --csv report/github.csv --patterns-repo patterns-repository --out-dir github_dataset

echo "Ontology and Ontology Design Pattern statistics ..."
python3 odp_stats.py --base github_dataset --out-dir github_dataset_stats --all-backends
python3 odp_stats.py --base Ontologies_reusingODPs --out-dir Ontologies_reusingODPs_stats --all-backends

echo "Building gold standard datasets ..."
backends=("hybrid" "char" "word" "sbert" "exact")
for backend in "${backends[@]}"; do
  python3 gold_dataset.py --base github_dataset --out-dir github_dataset_out --sim-backend "$backend" --materialize symlink --dataset-percents "100"
  python3 gold_dataset.py --base Ontologies_reusingODPs --out-dir Ontologies_reusingODPs_out --sim-backend "$backend" --materialize symlink --dataset-percents "100"
done
python3 gold_dataset.py --base github_dataset --out-dir github_dataset_out --sim-backend exact --materialize symlink --dataset-percents "100" --make-graph-100
python3 gold_dataset.py --base Ontologies_reusingODPs --out-dir Ontologies_reusingODPs_out --sim-backend exact --materialize symlink --dataset-percents "100" --make-graph-100

