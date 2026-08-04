#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_clusterstats.sh -- reproduce every table under results/iri/clusterstats/.
#
# These are the cluster-level statistics that rq5_per_ontology.jsonl cannot
# answer on its own: the runner stores the SELECTED hyper-parameter and the
# resulting P/R/F1, but not the number of clusters, their sizes, the noise share
# or the split/merge structure. Stages 2-4 therefore re-run the clustering.
#
# IMPORTANT -- the re-run is NOT the published process. gensim's Word2Vec uses
# hashfxn=hash, and Python salts string hashing per process, so the embeddings
# depend on PYTHONHASHSEED, which the published run did not fix. Same process =>
# identical; same PYTHONHASHSEED => identical; different PYTHONHASHSEED => not.
# Every stage below pins PYTHONHASHSEED, and stage 5 measures what that pinning
# is worth. For K-Means the published cluster counts ARE exact (config kmeans_kN
# has N clusters, checked in pred_count_identity_check.csv); for HDBSCAN the
# published counts are unrecoverable and the re-run is the estimate.
#
# Run from use-cases/odp-extraction/.  Total wall time on 8 cores: ~1.5 h.
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="${DATA_ROOT:-C:/Users/eno/Documents/PhD/MultiSource_server}"
MANIFEST="../../pipeline/out/MANIFEST.csv"
PUB="results/iri/rq5_per_ontology.jsonl"
OUT="results/iri/clusterstats"
RAW="$OUT/raw"
NSHARDS="${NSHARDS:-4}"
mkdir -p "$RAW"

# 1. ground-truth overlap and the hard-partition ceiling (no embedding needed)
python tools/gt_overlap_stats.py \
    --manifest "$MANIFEST" --data-root "$ROOT" \
    --datasets github noOBO --subset reuse --out "$OUT"

#    the companion iri+label track: different manifest, plus the adjudicated
#    label matches, exactly as the iri+label extraction run was scored
python tools/gt_overlap_stats.py \
    --manifest ../../pipeline/out_iri_label/MANIFEST.csv --data-root "$ROOT" \
    --datasets github noOBO --subset reuse \
    --approved-labels ../../pipeline/out/adjudication/human_sample.csv \
    --out "$OUT" --tag iri-label

# 2. cluster diagnostics over the full hyper-parameter grid, sharded
for s in $(seq 0 $((NSHARDS - 1))); do
    PYTHONHASHSEED=0 python tools/cluster_diagnostics.py \
        --manifest "$MANIFEST" --data-root "$ROOT" \
        --datasets github noOBO --subset reuse \
        --out "$RAW/diag_seed0.jsonl" --shard "$s" --n-shards "$NSHARDS" \
        > "$RAW/log_seed0_shard$s.txt" 2>&1 &
done
wait

# 3. two further hash seeds on one shard, for the seed-sensitivity table.
#    seed_sensitivity.py intersects the seeds on their common ontologies, so this
#    stage may be interrupted -- the shipped t7 table was produced from the first
#    29 ontologies of shard 1, which is what the n column reports. Let it run to
#    completion for the full 123.
for s in 1 2; do
    PYTHONHASHSEED=$s python tools/cluster_diagnostics.py \
        --manifest "$MANIFEST" --data-root "$ROOT" \
        --datasets github noOBO --subset reuse \
        --out "$RAW/diag_seed$s.jsonl" --shard 1 --n-shards "$NSHARDS" \
        > "$RAW/log_seed${s}_shard1.txt" 2>&1 &
done
wait

# 4. aggregate: T0-T6, T9 + the per-ontology tidy file
python tools/aggregate_cluster_stats.py \
    --published-jsonl "$PUB" \
    --diagnostics "$RAW/diag_seed0.jsonl.shard*" \
    --pairs-csv ../../pipeline/out/rq_iri_gated/pairs.csv \
    --manifest-csv "$MANIFEST" \
    --gt-overlap "$OUT/gt_overlap_per_ontology.csv" \
    --out "$OUT"

#    companion track: only the published-config statistics are available
python tools/aggregate_cluster_stats.py \
    --published-jsonl results/iri-label/rq5_per_ontology.jsonl \
    --pairs-csv ../../pipeline/out/rq_iri-label_gated/pairs.csv \
    --manifest-csv ../../pipeline/out_iri_label/MANIFEST.csv \
    --out "$OUT" --tag iri-label

# 5. seed sensitivity (T7)
python tools/seed_sensitivity.py \
    --dump "0=$RAW/diag_seed0.jsonl.shard1" \
    --dump "1=$RAW/diag_seed1.jsonl.shard1" \
    --dump "2=$RAW/diag_seed2.jsonl.shard1" \
    --out "$OUT"

# 6. null baselines (T8)
python tools/null_baseline.py \
    --diagnostics "$RAW/diag_seed0.jsonl.shard*" \
    --gt-overlap "$OUT/gt_overlap_per_ontology.csv" \
    --out "$OUT" --reps 100

echo "[done] tables in $OUT"
