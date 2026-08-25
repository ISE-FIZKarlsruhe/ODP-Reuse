#!/usr/bin/env bash
# Re-run both anonymised evaluations now that input paths are resolved.
# Before the fix, a relative --data-root pushed each ontology path past
# Windows' 260-character limit, directory listings came back empty, and 18
# GitHub ontologies were recorded as "no_ontology_file". They are present in
# the anonymised corpus; the runner could not see them.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
M=../../pipeline/out/MANIFEST.csv
C=../../release-iri/corpus-hashed
echo "[1/2] methods on the anonymised corpus"
python -u tools/run_method_eval.py --manifest "$M" --data-root "$C" \
  --corpus-layout release --out results/methods-iri-anon \
  --datasets github noOBO --subset reuse --corpus-name anonymised
echo "[2/2] embeddings on the anonymised corpus"
python -u tools/run_extraction_eval.py --manifest "$M" --data-root "$C" \
  --corpus-layout release --out results/iri-anon \
  --datasets github noOBO --subset reuse
echo "DONE"
