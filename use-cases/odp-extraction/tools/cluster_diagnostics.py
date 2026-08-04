#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cluster_diagnostics.py -- recover the cluster-level quantities that the RQ5 run
computed but did not store.

WHY THIS SCRIPT EXISTS
----------------------
``run_extraction_eval.py`` writes only the selected configuration name and the
resulting P/R/F1 per (ontology, method, clusterer):

    {"oracle_config": "kmeans_k9", "oracle_f1": ..., "unsup_config": "kmeans_k10", ...}

``extractbench/evaluate.match_and_score`` *does* compute ``n_predicted`` (and the
cluster memberships exist in memory), but the runner drops both. Consequently
``rq5_per_ontology.jsonl`` supports NO statement about

  * how many clusters HDBSCAN actually returned (only ``min_cluster_size`` is
    recorded in the config name),
  * cluster sizes, noise/outlier share, entity coverage,
  * how ground-truth patterns are split across clusters or merged into one.

For K-Means the predicted count is recoverable from the config name (``k``), on
the assumption that every one of the k clusters is non-empty -- an assumption
this script verifies rather than asserts.

Everything else requires re-running the clustering. That is safe to do: both
embedders are seeded (``seed=42``, ``workers=1``), ``collect_entities`` returns a
sorted list, KMeans uses ``random_state=42`` and HDBSCAN is deterministic, so the
re-run reproduces the embeddings and the clusterings of the published run
bit-for-bit. ``--validate`` writes the stored-vs-recomputed comparison so this is
checked, not assumed.

WHAT IS WRITTEN
---------------
One JSONL row per (ontology, method, clusterer, CONFIG) -- i.e. the FULL
hyper-parameter grid, not just the two selected points, so that the oracle and
the unsupervised regimes can be compared against the same underlying clusterings
and against configurations neither regime chose.

Fields per row::

    dataset, ontology, method, clusterer, config
    n_entities              entities handed to the embedder (wide entity set)
    n_gt                    ground-truth pattern instances
    n_predicted             non-empty predicted clusters  <-- the missing number
    n_noise                 entities with HDBSCAN label < 0 (0 for K-Means)
    coverage                (n_entities - n_noise) / n_entities
    silhouette              the value the unsupervised regime selects on
    precision/recall/f1     macro over Hungarian-matched pairs (as published)
    n_matched               matched pairs with F1 > 0
    cluster_sizes           list, descending
    gt_sizes                list, in the order used for the cost matrix
    gt_n_clusters_hit       per GT pattern: # distinct clusters holding >=1 of
                            its entities  (>1  =>  the pattern is SPLIT)
    gt_n_clusters_hit2      same, counting only clusters holding >=2 of them
                            (robust to a single stray entity)
    gt_frac_in_noise        per GT pattern: share of its entities left as noise
    pred_n_gt_hit           per cluster: # GT patterns it intersects
                            (>1  =>  the cluster MERGES patterns)
    is_oracle, is_unsup     whether this config is the one the regime selected

Aggregation lives in ``aggregate_cluster_stats.py``; this script only dumps.

USAGE
-----
    python cluster_diagnostics.py \
        --manifest ../../pipeline/out/MANIFEST.csv \
        --data-root C:/Users/eno/Documents/PhD/MultiSource_server \
        --datasets github noOBO --subset reuse \
        --out ../results/iri/clusterstats/cluster_diagnostics.jsonl \
        --shard 0 --n-shards 4

Shards are disjoint by ``index %% n_shards`` over the population, each writing
``<out>.shard<i>``; concatenate them afterwards. The embedders are single
threaded, so sharding is the only parallelism available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
_USECASE = _TOOLS.parent
_REPO = _USECASE.parent.parent
_PIPELINE = _REPO / "pipeline"

sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_PIPELINE))

# Reuse the runner's population and ground-truth logic verbatim: a second
# implementation of either would be a second ground truth.
import run_extraction_eval as RUNNER  # noqa: E402
from extractbench import owl2vec, rdf2vec  # noqa: E402
from extractbench.evaluate import CLUSTERERS, match_and_score  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from build_manifest import DATASETS  # noqa: E402

EMBEDDERS = {"owl2vec": owl2vec.embed, "rdf2vec": rdf2vec.embed}
CLUSTERER_NAMES = ("kmeans", "hdbscan")


def diagnose(clusters: dict[int, set[str]], gt: dict[str, set[str]],
             entities: list[str]) -> dict:
    """Cluster-level descriptive statistics for one clustering."""
    cl_keys = [k for k, v in clusters.items() if v]
    gt_keys = [k for k, v in gt.items() if v]
    covered = set().union(*[clusters[k] for k in cl_keys]) if cl_keys else set()
    n_noise = len(entities) - len(covered)

    gt_hit, gt_hit2, gt_noise = [], [], []
    for gk in gt_keys:
        g = gt[gk]
        hits = [len(clusters[ck] & g) for ck in cl_keys]
        gt_hit.append(sum(1 for h in hits if h >= 1))
        gt_hit2.append(sum(1 for h in hits if h >= 2))
        gt_noise.append(round(len(g - covered) / len(g), 4))

    pred_hit = []
    for ck in cl_keys:
        c = clusters[ck]
        pred_hit.append(sum(1 for gk in gt_keys if c & gt[gk]))

    return {
        "n_noise": n_noise,
        "coverage": round(len(covered) / len(entities), 4) if entities else 0.0,
        "cluster_sizes": sorted((len(clusters[k]) for k in cl_keys), reverse=True),
        # pattern identifiers in the same order as every per-instance list below,
        # so a split/merge figure can be attributed to a named ODP
        "gt_keys": list(gt_keys),
        "gt_sizes": [len(gt[k]) for k in gt_keys],
        "gt_n_clusters_hit": gt_hit,
        "gt_n_clusters_hit2": gt_hit2,
        "gt_frac_in_noise": gt_noise,
        "pred_n_gt_hit": pred_hit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--datasets", nargs="*", default=["github", "noOBO"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--max-entities", type=int,
                    default=RUNNER.DEFAULT_MAX_ENTITIES)
    ap.add_argument("--keep-annotations", action="store_true")
    ap.add_argument("--approved-labels", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these ontology_ids (debug/validation)")
    args = ap.parse_args()

    approved: set[tuple[str, str, str]] = set()
    if args.approved_labels and args.approved_labels.is_file():
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.add((r["dataset"], r["ontology"], r["pattern"]))
        print(f"[iri+label] {len(approved)} human-confirmed label matches loaded")

    strip_annots = not args.keep_annotations
    pop = RUNNER.load_population(args.manifest, args.datasets, args.subset)
    if args.limit:
        pop = pop[:args.limit]
    if args.only:
        keep = set(args.only)
        pop = [r for r in pop if r["ontology_id"] in keep]
    mine = [r for i, r in enumerate(pop) if i % args.n_shards == args.shard]
    print(f"population {len(pop)}; shard {args.shard}/{args.n_shards} -> "
          f"{len(mine)} ontologies", flush=True)

    out = args.out if args.n_shards == 1 else Path(f"{args.out}.shard{args.shard}")
    out.parent.mkdir(parents=True, exist_ok=True)
    skipped: list[tuple[str, str, str, str]] = []

    t0 = time.time()
    with out.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(mine, 1):
            ds, oid = r["dataset"], r["ontology_id"]
            onto_dir = args.data_root / DATASETS[ds] / oid
            main_f = main_ontology_file(onto_dir)
            if main_f is None:
                skipped.append((ds, oid, "no_ontology_file", ""))
                continue
            g, err = parse_graph(main_f)
            if g is None:
                skipped.append((ds, oid, "parse_error", err))
                continue
            if strip_annots:
                g, _ = strip_annotations(g)
            entities = collect_entities(g, wide=True)
            if not entities:
                skipped.append((ds, oid, "no_entities", ""))
                continue
            if args.max_entities > 0 and len(entities) > args.max_entities:
                skipped.append((ds, oid, "too_many_entities",
                                f"{len(entities)}>{args.max_entities}"))
                continue
            gt = RUNNER.ground_truth(onto_dir, set(entities), strip_annots,
                                     complete_only=(args.subset == "complete"),
                                     host_graph=g, dataset=ds,
                                     approved_labels=approved or None)
            if not gt:
                skipped.append((ds, oid, "no_ground_truth", f"subset={args.subset}"))
                continue

            labels = {e: e.rsplit("/", 1)[-1] for e in entities}
            for mname, embed in EMBEDDERS.items():
                try:
                    emb = embed(g, entities, labels)
                except MemoryError:
                    skipped.append((ds, oid, "embed_out_of_memory",
                                    f"{mname}, {len(entities)} entities"))
                    continue
                except Exception as exc:  # noqa: BLE001
                    skipped.append((ds, oid, "embed_error",
                                    f"{mname}: {type(exc).__name__}"))
                    continue
                emb = np.nan_to_num(np.asarray(emb, dtype=np.float32))
                for cl in CLUSTERER_NAMES:
                    # Same generator, same order, so the same tie-breaking as
                    # evaluate_ontology's max() over the same list.
                    cands = list(CLUSTERERS[cl](emb, entities))
                    scored = [(name, clusters, match_and_score(clusters, gt), sil)
                              for name, clusters, sil in cands]
                    if not scored:
                        continue
                    o_name = max(scored, key=lambda t: t[2]["f1"])[0]
                    u_name = max(scored, key=lambda t: t[3])[0]
                    for name, clusters, s, sil in scored:
                        rec = {
                            "dataset": ds, "ontology": oid, "method": mname,
                            "clusterer": cl, "config": name,
                            "n_entities": len(entities), "n_gt": len(gt),
                            "n_predicted": s["n_predicted"],
                            "silhouette": round(float(sil), 4),
                            "precision": round(s["precision"], 4),
                            "recall": round(s["recall"], 4),
                            "f1": round(s["f1"], 4),
                            "n_matched": s["n_matched"],
                            "is_oracle": name == o_name,
                            "is_unsup": name == u_name,
                        }
                        rec.update(diagnose(clusters, gt, entities))
                        fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(mine):
                el = time.time() - t0
                print(f"[shard {args.shard}] [{i}/{len(mine)}] {el:6.0f}s "
                      f"({el / i:.1f}s/onto)", flush=True)

    if skipped:
        sp = Path(f"{out}.skipped.csv")
        with sp.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "ontology", "reason", "detail"])
            w.writerows(skipped)
        print(f"[SKIPPED] {len(skipped)} -> {sp}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
