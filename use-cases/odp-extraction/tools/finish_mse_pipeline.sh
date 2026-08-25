#!/usr/bin/env bash
# finish_mse_pipeline.sh -- run the remaining MSE work end to end, in order.
#
# The conceptual-components pipeline needs three stages that cannot share a
# process: documents (Windows, on the corpus), disambiguation (inside WSL, beside
# the Framester/UKB server, because the server hard-codes its bind address to
# localhost), and scoring (Windows again). Chaining them here means the whole
# remainder is one job that either finishes or fails visibly, rather than a
# sequence someone has to babysit.
#
# Steps:
#   1. wait for the documents stage to produce documents.json
#   2. disambiguate every community against the step-3 server, inside WSL
#   3. cluster and score (stage 6-7), emitting predictions
#   4. regenerate the statistics over ALL prediction sets
#   5. error-analyse each run
#   6. rebuild the release with every prediction file present
#
# Usage:  ./tools/finish_mse_pipeline.sh [release-version]

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="${1:-1.0.0}"
UC_WSL="/mnt/c/Users/eno/Documents/PhD/odp-reuse-benchmark/use-cases/odp-extraction"
TL="C:/Users/eno/AppData/Local/Temp/claude/c--Users-eno-Documents-PhD-odp-reuse-benchmark/44cac895-115a-4a12-a9c1-42db1ff2f7e5/scratchpad/conceptual-components/conceptual-components-extraction/project_root/data/topLevelAnchestors-prefix.csv"
LOG="results/mseo-cc-full/finish.log"
mkdir -p results/mseo-cc-full
: > "$LOG"
say() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "=== 1/6  waiting for the documents stage ==="
for _ in $(seq 1 240); do
  [[ -s results/mseo-cc-full/documents.json ]] && break
  sleep 15
done
if [[ ! -s results/mseo-cc-full/documents.json ]]; then
  say "FAILED: documents.json never appeared. Run --stage documents first."
  exit 1
fi
N=$(python -c "import json;print(len(json.load(open('results/mseo-cc-full/documents.json',encoding='utf-8'))))")
say "documents.json ready: $N communities"

say "=== 2/6  disambiguation inside WSL (this is the long one) ==="
if ! wsl -d Ubuntu -- bash -lc "ss -ltn 2>/dev/null | grep -q 8080"; then
  say "FAILED: no Framester/UKB server on localhost:8080 inside WSL."
  say "  Start it with:  cd ~/cc/run/disambiguation_framedetection && \\"
  say "    setsid bash -c 'tail -f /dev/null | java -jar ../lgu.commons.server.jar' &"
  exit 1
fi
wsl -d Ubuntu -- bash -lc "cd $UC_WSL && python3 tools/framester_client.py \
  --in results/mseo-cc-full/documents.json \
  --out results/mseo-cc-full/synsets.json --workers 10 --resume" 2>&1 | tee -a "$LOG"
[[ -s results/mseo-cc-full/synsets.json ]] || { say "FAILED: no synsets.json"; exit 1; }
say "disambiguation done"

say "=== 3/6  cluster and score (steps 5-7) ==="
python -u tools/run_asprino_full.py --corpus corpora/mseo \
  --out results/mseo-cc-full --top-level "$TL" \
  --stage score --resume --corpus-name mseo 2>&1 | tee -a "$LOG"

say "=== 4/6  statistics over every prediction set ==="
PREDS=(results/mseo/predictions.jsonl results/mseo-embed/predictions.jsonl)
[[ -s results/mseo-cc-full/predictions.jsonl ]] && PREDS+=(results/mseo-cc-full/predictions.jsonl)
say "  using: ${PREDS[*]}"
python -u tools/mse_stats.py --predictions "${PREDS[@]}" --out stats/mseo \
  --patterns-root "C:/Users/eno/Documents/PhD/MultiSource_server/patterns-repository" 2>&1 | tee -a "$LOG"

say "=== 5/6  error analysis ==="
for r in results/mseo results/mseo-embed results/mseo-cc-full; do
  [[ -d "$r" ]] || continue
  say "--- $r"
  python tools/error_analysis.py --run "$r" 2>&1 | tail -20 | tee -a "$LOG"
done

say "=== 6/6  rebuild the release ==="
python -u tools/build_mse_release.py --version "$VERSION" \
  --corpus corpora/mseo --results results --stats stats/mseo \
  --out ../../mse-release 2>&1 | tee -a "$LOG"
(cd ../../mse-release && python validate_release.py) 2>&1 | tee -a "$LOG"

say "=== DONE ==="
say "release: mse-release/ (v$VERSION)"
python - <<'EOF' 2>&1 | tee -a "$LOG"
import json, glob, os
for p in sorted(glob.glob('../../mse-release/predictions/*.jsonl')):
    n = sum(1 for _ in open(p, encoding='utf-8'))
    print(f"  {os.path.basename(p):38s} {n:6d} rows")
EOF
