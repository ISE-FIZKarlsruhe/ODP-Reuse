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
# Comprehensive IRI extraction (single Python run via heredoc)
# -----------------------------
TMP_IRIS="$(mktemp)"
trap 'rm -f "$TMP_IRIS"' EXIT

python3 - "$IRI_HOST_FILTER" "$EXTRACT_REPORT" "$DIRS_SUMMARY" "$DIRS_NO_IRI" "$FILES_NO_IRI" "${FILES[@]}" > "$TMP_IRIS" <<'PY'
import sys, io, os, re, csv
from collections import defaultdict

host_filter = sys.argv[1]
EXTRACT_REPORT = sys.argv[2]
DIRS_SUMMARY   = sys.argv[3]
DIRS_NO_IRI    = sys.argv[4]
FILES_NO_IRI   = sys.argv[5]
files = sys.argv[6:]

# Regexes
RX_XML_BASE       = re.compile(r'xml:base\s*=\s*"([^"]+)"', re.I)
RX_XMLNS_BASE     = re.compile(r'xmlns:base\s*=\s*"([^"]+)"', re.I)   # non-standard but used here
RX_XMLNS_DEFAULT  = re.compile(r'xmlns\s*=\s*"([^"]+)"', re.I)        # default namespace (strip '#')
RX_ONTO_ATTR      = re.compile(r'<owl:Ontology[^>]*?(?:IRI|rdf:about)\s*=\s*"([^"]+)"', re.I)
RX_OWLXML_ONTOIRI = re.compile(r'<Ontology[^>]*?\sontologyIRI\s*=\s*"([^"]+)"', re.I)   # OWL/XML
RX_TTL_AT_BASE    = re.compile(r'@base\s*<([^>]+)>', re.I)
RX_TTL_BASE       = re.compile(r'^\s*BASE\s*<([^>]+)>\s*$', re.I|re.M)                 # Turtle 1.1
RX_TTL_EXPL_SUBJ  = re.compile(r'^\s*<([^>]+)>\s+a\s+owl:Ontology\b', re.I|re.M)
RX_TTL_VER_IRI    = re.compile(r'owl:versionIRI\s*<([^>]+)>', re.I)
RX_TTL_PREFIX_COLON = re.compile(r'@prefix\s*:\s*<([^>]+)>\s*\.', re.I)
RX_TTL_COLON_IS_ONTO= re.compile(r'^\s*:\s+a\s+owl:Ontology\b', re.I|re.M)

def read_text(p):
    try:
        with io.open(p, 'r', encoding='utf-8', errors='ignore') as fh:
            return fh.read()
    except Exception:
        return ""

def keep(iri: str) -> bool:
    return (iri and (host_filter == "" or host_filter.lower() in iri.lower()))

def add_match(matches, iri, method):
    iri = iri.strip()
    if iri and keep(iri):
        matches.add((iri, method))

rows = []
dir_counts = defaultdict(lambda: {"total":0, "with":0, "without":0, "iris":set()})
files_without = []

for p in files:
    text = read_text(p)
    d = os.path.dirname(p)
    dir_counts[d]["total"] += 1
    matches = set()

    for m in RX_XML_BASE.findall(text):
        add_match(matches, m, "xml:base")
    for m in RX_XMLNS_BASE.findall(text):
        add_match(matches, m, "xmlns:base (namespace)")
    for m in RX_XMLNS_DEFAULT.findall(text):
        iri = m.strip().rstrip('#')  # e.g., http://.../Airline.owl#
        add_match(matches, iri, "xmlns (default namespace)")
    for m in RX_ONTO_ATTR.findall(text):
        add_match(matches, m, "owl:Ontology IRI/rdf:about")
    for m in RX_OWLXML_ONTOIRI.findall(text):
        add_match(matches, m, "OWL/XML ontologyIRI")
    for m in RX_TTL_AT_BASE.findall(text):
        add_match(matches, m, "Turtle @base")
    for m in RX_TTL_BASE.findall(text):
        add_match(matches, m, "Turtle BASE")
    for m in RX_TTL_EXPL_SUBJ.findall(text):
        add_match(matches, m, "Turtle <IRI> a owl:Ontology")
    for m in RX_TTL_VER_IRI.findall(text):
        add_match(matches, m, "owl:versionIRI")

    # Heuristic: default ':' prefix + ': a owl:Ontology'
    m = RX_TTL_PREFIX_COLON.search(text)
    if m and RX_TTL_COLON_IS_ONTO.search(text):
        add_match(matches, m.group(1).strip(), "Turtle @prefix : <> + : a owl:Ontology")

    if matches:
        dir_counts[d]["with"] += 1
        for iri, method in sorted(matches):
            rows.append({"file": p, "dir": d, "iri": iri, "method": method})
            dir_counts[d]["iris"].add(iri)
    else:
        dir_counts[d]["without"] += 1
        files_without.append(p)

# Detailed per-file report
with open(EXTRACT_REPORT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["file","dir","iri","method"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

# Per-directory summary
with open(DIRS_SUMMARY, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["dir","total_files","files_with_iri","files_without_iri","unique_iris"])
    w.writeheader()
    for d in sorted(dir_counts):
        w.writerow({
            "dir": d,
            "total_files": dir_counts[d]["total"],
            "files_with_iri": dir_counts[d]["with"],
            "files_without_iri": dir_counts[d]["without"],
            "unique_iris": len(dir_counts[d]["iris"]),
        })

# Dirs/files with no IRI
with open(DIRS_NO_IRI, "w", encoding="utf-8") as f:
    for d in sorted(dir_counts):
        if dir_counts[d]["with"] == 0:
            f.write(d + "\n")
with open(FILES_NO_IRI, "w", encoding="utf-8") as f:
    for p in files_without:
        f.write(p + "\n")

# Print unique IRIs to stdout (captured by the shell into $TMP_IRIS)
uniq = sorted({r["iri"] for r in rows})
for iri in uniq:
    print(iri)
PY

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

run: all major sources (auto-discovers ODP IRIs)
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

