#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gt_overlap_stats.py -- do the GROUND-TRUTH ODP instances of one ontology overlap?

This bounds what any hard clustering can achieve, and it needs no embedding: it
re-derives the same ground truth as ``run_extraction_eval.py`` (same
``ground_truth()`` function, same population filter) and then measures, per
ontology, whether two reused patterns share host entities.

Per-ontology output (``gt_overlap_per_ontology.csv``)::

    dataset, ontology, n_entities, n_gt
    gt_sizes                        ';'-joined pattern instance sizes
    n_gt_entities_union             |union of all GT instances|
    n_gt_entities_multi             entities in >= 2 GT instances  ("contested")
    frac_gt_entities_multi          contested / union
    gt_entity_share_of_ontology     union / n_entities
    n_pairs                         m(m-1)/2 pattern pairs
    n_pairs_overlapping             pairs sharing >= 1 entity
    n_pairs_nested                  pairs where one instance contains the other
    max_pair_jaccard                worst-case pair Jaccard
    mean_pair_jaccard
    partition_ceiling_f1            EXACT best macro-F1 attainable by a hard
                                    partition that covers all m patterns
                                    (== 1.0 iff the instances are disjoint)
    ceiling_method                  'disjoint' or 'exact_assignment'

CEILING
-------
The published metric matches predicted clusters to GT instances by Hungarian
assignment and macro-averages F1 over the matched pairs. A hard partition can
give a contested entity to only one cluster, so overlapping GT instances make
F1 = 1 unattainable *for a clustering that covers every instance*. The exact
optimum is computed, not approximated: with clusters restricted to subsets of
their matched instance (extra entities only lower precision), the problem is to
choose disjoint ``s_i <= g_i`` maximising ``mean_i 2|s_i| / (|s_i| + |g_i|)``.
Each ``f_i`` is concave and increasing in ``|s_i|``, so expanding every instance
into unit slots with decreasing marginal gains turns it into a bipartite
max-weight assignment between contested entities and slots, whose optimum equals
the optimum of the original concave problem.

A SEPARATE CAVEAT, reported by this script but not a property of the data:
``match_and_score`` averages over ``min(n_predicted, n_gt)`` matched pairs, so
ground-truth instances left unmatched when ``n_predicted < n_gt`` are dropped
from the average rather than scored as misses. Under that convention the nominal
ceiling is 1.0 for every ontology -- a single cluster equal to the easiest
instance suffices. ``partition_ceiling_f1`` therefore deliberately reports the
ceiling under the stricter, meaningful requirement that all instances be covered.

USAGE
-----
    python gt_overlap_stats.py \
        --manifest ../../pipeline/out/MANIFEST.csv \
        --data-root C:/Users/eno/Documents/PhD/MultiSource_server \
        --datasets github noOBO --subset reuse \
        --out ../results/iri/clusterstats
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent
_USECASE = _TOOLS.parent
_REPO = _USECASE.parent.parent
_PIPELINE = _REPO / "pipeline"
sys.path.insert(0, str(_TOOLS))
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_PIPELINE))

import run_extraction_eval as RUNNER  # noqa: E402
from odpqual.entities import collect_entities, parse_graph, strip_annotations  # noqa: E402
from odpqual.gate import main_ontology_file  # noqa: E402
from build_manifest import DATASETS  # noqa: E402


def partition_ceiling(gt: dict[str, set[str]]) -> tuple[float, str]:
    """Exact best macro-F1 of a hard partition that covers every GT instance."""
    keys = [k for k, v in gt.items() if v]
    sets = [gt[k] for k in keys]
    counts = Counter(e for s in sets for e in s)
    contested = [e for e, c in counts.items() if c >= 2]
    if not contested:
        return 1.0, "disjoint"

    def f(s: int, n: int) -> float:
        return 0.0 if s == 0 else 2.0 * s / (s + n)

    base = [len(s) - sum(1 for e in contested if e in s) for s in sets]
    ncon = [sum(1 for e in contested if e in s) for s in sets]

    # slot j of instance i = the j-th contested entity handed to i
    slots: list[tuple[int, float]] = []
    for i, s in enumerate(sets):
        for j in range(1, ncon[i] + 1):
            slots.append((i, f(base[i] + j, len(s)) - f(base[i] + j - 1, len(s))))

    BIG = 1e6
    cost = np.full((len(contested), len(slots)), BIG)
    for r, e in enumerate(contested):
        for c, (i, gain) in enumerate(slots):
            if e in sets[i]:
                cost[r, c] = -gain
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(cost)
    used = Counter()
    for r, c in zip(rows, cols):
        if cost[r, c] < BIG / 2:
            used[slots[c][0]] += 1
    tot = sum(f(base[i] + used[i], len(sets[i])) for i in range(len(sets)))
    return round(tot / len(sets), 6), "exact_assignment"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--from-csv", type=Path, default=None,
                    help="re-derive the summary from an existing "
                         "gt_overlap_per_ontology.csv instead of re-parsing the "
                         "corpus (the per-ontology file is the primary record)")
    ap.add_argument("--datasets", nargs="*", default=["github", "noOBO"])
    ap.add_argument("--subset", choices=("complete", "reuse"), default="reuse")
    ap.add_argument("--max-entities", type=int, default=RUNNER.DEFAULT_MAX_ENTITIES)
    ap.add_argument("--keep-annotations", action="store_true")
    ap.add_argument("--approved-labels", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        rows = list(csv.DictReader(args.from_csv.open(encoding="utf-8")))
        for r in rows:
            for k in ("n_entities", "n_gt", "n_gt_entities_union",
                      "n_gt_entities_multi", "n_pairs", "n_pairs_overlapping",
                      "n_pairs_nested"):
                r[k] = int(r[k])
            for k in ("frac_gt_entities_multi", "gt_entity_share_of_ontology",
                      "max_pair_jaccard", "mean_pair_jaccard",
                      "partition_ceiling_f1"):
                r[k] = float(r[k])
        summarise(rows, args.out, args.tag)
        return
    if not args.manifest or not args.data_root:
        raise SystemExit("--manifest and --data-root are required unless --from-csv")

    approved: set[tuple[str, str, str]] = set()
    if args.approved_labels and args.approved_labels.is_file():
        with args.approved_labels.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("human_verdict", "").strip().lower() == "reuse":
                    approved.add((r["dataset"], r["ontology"], r["pattern"]))

    strip_annots = not args.keep_annotations
    pop = RUNNER.load_population(args.manifest, args.datasets, args.subset)
    if args.limit:
        pop = pop[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i, r in enumerate(pop, 1):
        ds, oid = r["dataset"], r["ontology_id"]
        onto_dir = args.data_root / DATASETS[ds] / oid
        main_f = main_ontology_file(onto_dir)
        if main_f is None:
            continue
        g, err = parse_graph(main_f)
        if g is None:
            continue
        if strip_annots:
            g, _ = strip_annotations(g)
        ents = collect_entities(g, wide=True)
        if not ents:
            continue
        if args.max_entities > 0 and len(ents) > args.max_entities:
            continue
        gt = RUNNER.ground_truth(onto_dir, set(ents), strip_annots,
                                 complete_only=(args.subset == "complete"),
                                 host_graph=g, dataset=ds,
                                 approved_labels=approved or None)
        if not gt:
            continue
        keys = sorted(gt)
        sets = [gt[k] for k in keys]
        union = set().union(*sets)
        counts = Counter(e for s in sets for e in s)
        multi = [e for e, c in counts.items() if c >= 2]
        jac, n_ov, n_nest = [], 0, 0
        for a, b in itertools.combinations(range(len(sets)), 2):
            inter = len(sets[a] & sets[b])
            uni = len(sets[a] | sets[b])
            jac.append(inter / uni if uni else 0.0)
            if inter:
                n_ov += 1
            if sets[a] <= sets[b] or sets[b] <= sets[a]:
                n_nest += 1
        ceil, meth = partition_ceiling(gt)
        rows.append({
            "dataset": ds, "ontology": oid, "n_entities": len(ents),
            "n_gt": len(gt),
            "gt_sizes": ";".join(str(len(s)) for s in sets),
            "n_gt_entities_union": len(union),
            "n_gt_entities_multi": len(multi),
            "frac_gt_entities_multi": round(len(multi) / len(union), 4) if union else 0.0,
            "gt_entity_share_of_ontology": round(len(union) / len(ents), 4),
            "n_pairs": len(jac),
            "n_pairs_overlapping": n_ov,
            "n_pairs_nested": n_nest,
            "max_pair_jaccard": round(max(jac), 4) if jac else 0.0,
            "mean_pair_jaccard": round(float(np.mean(jac)), 4) if jac else 0.0,
            "partition_ceiling_f1": ceil,
            "ceiling_method": meth,
        })
        if i % 50 == 0:
            print(f"[{i}/{len(pop)}]", flush=True)

    tag = f"_{args.tag}" if args.tag else ""
    p = args.out / f"gt_overlap_per_ontology{tag}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {p}  ({len(rows)} ontologies)")
    summarise(rows, args.out, args.tag)


def summarise(rows: list[dict], out: Path, tag_arg: str) -> None:
    """Portion-level summary.

    Overlap is only *possible* where an ontology reuses two or more patterns, so
    every overlap figure is reported twice: over all ontologies (the population
    the F1 table averages over) and over the multi-pattern ontologies alone (the
    population where the question is meaningful).
    """
    tag = f"_{tag_arg}" if tag_arg else ""
    summ: list[dict] = []
    for ds in sorted({r["dataset"] for r in rows}) + ["ALL"]:
        sub = rows if ds == "ALL" else [r for r in rows if r["dataset"] == ds]
        multi_onts = [r for r in sub if r["n_gt_entities_multi"] > 0]
        mgt = [r for r in sub if r["n_gt"] > 1]
        gt_all = [int(x) for r in sub for x in r["gt_sizes"].split(";")]
        summ.append({
            "portion": ds, "n_ontologies": len(sub),
            "n_ontologies_multi_gt": len(mgt),
            "mean_n_gt": round(float(np.mean([r["n_gt"] for r in sub])), 3),
            "median_n_gt": float(np.median([r["n_gt"] for r in sub])),
            "max_n_gt": max(r["n_gt"] for r in sub),
            "mean_gt_size": round(float(np.mean(gt_all)), 2),
            "median_gt_size": float(np.median(gt_all)),
            "p90_gt_size": float(np.percentile(gt_all, 90)),
            "n_ontologies_with_gt_overlap": len(multi_onts),
            "pct_ontologies_with_gt_overlap": round(100 * len(multi_onts) / len(sub), 2),
            "pct_multiGT_ontologies_with_gt_overlap": round(
                100 * len(multi_onts) / len(mgt), 2) if mgt else "",
            "n_pairs_total": sum(r["n_pairs"] for r in sub),
            "n_pairs_overlapping": sum(r["n_pairs_overlapping"] for r in sub),
            "pct_pairs_overlapping": round(
                100 * sum(r["n_pairs_overlapping"] for r in sub)
                / max(sum(r["n_pairs"] for r in sub), 1), 2),
            "n_pairs_nested": sum(r["n_pairs_nested"] for r in sub),
            "pct_pairs_nested": round(
                100 * sum(r["n_pairs_nested"] for r in sub)
                / max(sum(r["n_pairs"] for r in sub), 1), 2),
            "mean_frac_gt_entities_multi": round(
                float(np.mean([r["frac_gt_entities_multi"] for r in sub])), 4),
            "mean_frac_gt_entities_multi_multiGT": round(
                float(np.mean([r["frac_gt_entities_multi"] for r in mgt])), 4) if mgt else "",
            "mean_gt_entity_share_of_ontology": round(
                float(np.mean([r["gt_entity_share_of_ontology"] for r in sub])), 4),
            "median_gt_entity_share_of_ontology": round(
                float(np.median([r["gt_entity_share_of_ontology"] for r in sub])), 4),
            "mean_partition_ceiling_f1": round(
                float(np.mean([r["partition_ceiling_f1"] for r in sub])), 4),
            "mean_partition_ceiling_f1_multiGT": round(
                float(np.mean([r["partition_ceiling_f1"] for r in mgt])), 4) if mgt else "",
            "min_partition_ceiling_f1": round(
                min(r["partition_ceiling_f1"] for r in sub), 4),
            "n_ontologies_ceiling_below_0_9": sum(
                1 for r in sub if r["partition_ceiling_f1"] < 0.9),
        })
    p2 = out / f"gt_overlap_summary{tag}.csv"
    with p2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print(f"[OK] {p2}")
    for s in summ:
        print(f"  {s['portion']:8s} n={s['n_ontologies']:4d} "
              f"multiGT={s['n_ontologies_multi_gt']:4d} "
              f"overlap_onts={s['n_ontologies_with_gt_overlap']:4d} "
              f"({s['pct_multiGT_ontologies_with_gt_overlap']}% of multiGT) "
              f"ceiling={s['mean_partition_ceiling_f1']:.4f} "
              f"(multiGT {s['mean_partition_ceiling_f1_multiGT']})")


if __name__ == "__main__":
    main()
