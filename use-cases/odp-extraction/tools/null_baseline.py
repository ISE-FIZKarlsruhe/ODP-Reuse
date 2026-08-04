#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
null_baseline.py -- what does the reported F1 look like if the clusters carry no
information at all?

Three reference points, all scored with the published metric
(``extractbench.evaluate.match_and_score``: Hungarian matching on 1 - F1, macro
average over the matched pairs):

SIZE-MATCHED RANDOM RELABELLING  (the null for the UNSUPERVISED regime)
    Keep the selected clustering's exact size profile -- same number of clusters,
    same cluster sizes, same number of noise points -- and assign entities to
    those slots uniformly at random. Whatever the real clustering scores above
    this comes from *which* entities were grouped, not from the granularity of
    the grouping.

BEST-OF-GRID RANDOM RELABELLING  (the null for the ORACLE regime)
    The oracle regime reports ``max`` over the hyper-parameter grid of the
    observed F1, so its null must be given the same maximisation: per replicate,
    every configuration in the grid is relabelled at random and the best of them
    is kept. Comparing an oracle-selected score against a single-configuration
    null would overstate the method.

WHOLE-ONTOLOGY SINGLE CLUSTER
    One cluster containing every entity. Because the metric averages over
    ``min(n_predicted, n_gt)`` matched pairs, this scores
    ``max_i 2|g_i| / (|g_i| + |E|)`` exactly -- no simulation needed.

Evaluated only on ontologies whose ground-truth instances are pairwise DISJOINT
(``n_gt_entities_multi == 0`` in gt_overlap_per_ontology.csv): there a random
relabelling depends on the ground truth solely through the instance sizes, so the
simulation is exact rather than approximate. Ontologies with overlapping
instances are excluded and counted.

USAGE
-----
    python null_baseline.py \
        --diagnostics "../results/iri/clusterstats/raw/diag_seed0.jsonl.shard*" \
        --gt-overlap ../results/iri/clusterstats/gt_overlap_per_ontology.csv \
        --out ../results/iri/clusterstats --reps 100
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def macro_f1_from_counts(inter: np.ndarray, gt_sizes: np.ndarray,
                         cl_sizes: np.ndarray) -> float:
    """match_and_score's macro F1 expressed in intersection counts.

    ``inter[i, j] = |g_i & c_j|``. Empty clusters are dropped first, exactly as
    ``match_and_score`` drops them.
    """
    keep = cl_sizes > 0
    if not keep.any():
        return 0.0
    inter = inter[:, keep]
    cs = cl_sizes[keep]
    p = inter / cs[None, :]
    r = inter / gt_sizes[:, None]
    denom = p + r
    f1 = np.where(inter > 0, 2 * p * r / np.where(denom == 0, 1.0, denom), 0.0)
    rows, cols = linear_sum_assignment(1.0 - f1)
    return float(f1[rows, cols].mean())


def one_rep(labels: np.ndarray, starts: np.ndarray, gt: np.ndarray,
            cl_sizes: np.ndarray, rng: np.random.Generator) -> float:
    """One size-matched random relabelling, scored."""
    lab = rng.permutation(labels)
    k = cl_sizes.size
    inter = np.zeros((gt.size, k), dtype=float)
    for i, (s, sz) in enumerate(zip(starts, gt)):
        blk = lab[s:s + sz]
        blk = blk[blk >= 0]
        if blk.size:
            inter[i] = np.bincount(blk, minlength=k)
    return macro_f1_from_counts(inter, gt.astype(float), cl_sizes.astype(float))


def group_nulls(grid: list[dict], gt_sizes: list[int], reps: int,
                rng: np.random.Generator) -> tuple[dict[str, float], float]:
    """Per-config mean null F1, and the mean best-of-grid null F1."""
    gt = np.asarray(gt_sizes)
    starts = np.concatenate([[0], np.cumsum(gt)[:-1]])
    prepared = []
    for r in grid:
        if not r["cluster_sizes"]:
            prepared.append(None)
            continue
        cs = np.asarray(r["cluster_sizes"])
        labels = np.concatenate([np.repeat(np.arange(cs.size), cs),
                                 np.full(r["n_noise"], -1, dtype=int)])
        prepared.append((labels, cs))
    tot = np.zeros(len(grid))
    best = 0.0
    for _ in range(reps):
        vals = np.zeros(len(grid))
        for j, prep in enumerate(prepared):
            if prep is None:
                continue
            vals[j] = one_rep(prep[0], starts, gt, prep[1], rng)
        tot += vals
        best += float(vals.max())
    return ({r["config"]: float(t / reps) for r, t in zip(grid, tot)},
            best / reps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagnostics", required=True)
    ap.add_argument("--gt-overlap", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    disjoint: set[tuple[str, str]] = set()
    n_ov = 0
    for r in csv.DictReader(args.gt_overlap.open(encoding="utf-8")):
        if int(r["n_gt_entities_multi"]) == 0:
            disjoint.add((r["dataset"], r["ontology"]))
        else:
            n_ov += 1

    rows = []
    for f in sorted(glob.glob(args.diagnostics)):
        with open(f, encoding="utf-8") as fh:
            rows += [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if (r["dataset"], r["ontology"]) in disjoint]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["ontology"], r["method"], r["clusterer"])].append(r)
    print(f"{len(groups)} (ontology, model, clusterer) groups on disjoint-GT "
          f"ontologies; {n_ov} overlapping-GT ontologies excluded", flush=True)

    rng = np.random.default_rng(args.seed)
    per: list[dict] = []
    for gi, (key, grid) in enumerate(sorted(groups.items()), 1):
        ds, oid, m, cl = key
        gt_sizes = grid[0]["gt_sizes"]
        cfg_null, best_null = group_nulls(grid, gt_sizes, args.reps, rng)
        single = max(2 * s / (s + grid[0]["n_entities"]) for s in gt_sizes)
        for reg, flag in (("oracle", "is_oracle"), ("unsup", "is_unsup")):
            sel = [r for r in grid if r[flag]]
            if not sel:
                continue
            r = sel[0]
            per.append({
                "portion": ds, "ontology": oid, "model": m, "clusterer": cl,
                "regime": reg, "config": r["config"],
                "n_entities": r["n_entities"], "n_gt": r["n_gt"],
                "n_predicted": r["n_predicted"], "f1_observed": r["f1"],
                # oracle maximises over the grid, so its null must too
                "f1_null": round(best_null if reg == "oracle"
                                 else cfg_null[r["config"]], 4),
                "null_kind": ("best_of_grid_random" if reg == "oracle"
                              else "size_matched_random"),
                "f1_null_selected_config_only": round(cfg_null[r["config"]], 4),
                "f1_single_cluster": round(single, 4),
            })
        if gi % 200 == 0:
            print(f"  [{gi}/{len(groups)}]", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    p = args.out / f"t8_null_baseline_per_ontology{tag}.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    print(f"[OK] {p}  ({len(per)} rows)")

    summ: list[dict] = []
    portions = sorted({x["portion"] for x in per})
    for ds in portions + ["ALL"]:
        for m in ("owl2vec", "rdf2vec"):
            for cl in ("kmeans", "hdbscan"):
                for reg in ("oracle", "unsup"):
                    sub = [x for x in per
                           if (ds == "ALL" or x["portion"] == ds)
                           and x["model"] == m and x["clusterer"] == cl
                           and x["regime"] == reg]
                    if not sub:
                        continue
                    ob = np.array([x["f1_observed"] for x in sub])
                    nu = np.array([x["f1_null"] for x in sub])
                    sg = np.array([x["f1_single_cluster"] for x in sub])
                    summ.append({
                        "portion": ds, "model": m, "clusterer": cl, "regime": reg,
                        "null_kind": sub[0]["null_kind"],
                        "n_ontologies": len(sub),
                        "mean_f1_observed": round(float(ob.mean()), 4),
                        "mean_f1_null": round(float(nu.mean()), 4),
                        "mean_f1_single_cluster": round(float(sg.mean()), 4),
                        "lift_over_null": round(float(ob.mean() - nu.mean()), 4),
                        "ratio_over_null": round(float(ob.mean() / nu.mean()), 3)
                        if nu.mean() > 0 else "",
                        "pct_ontologies_at_or_below_null": round(
                            100 * float(np.mean(ob <= nu)), 2),
                        "pct_ontologies_at_or_below_single_cluster": round(
                            100 * float(np.mean(ob <= sg)), 2),
                    })
    p2 = args.out / f"t8_null_baseline_summary{tag}.csv"
    with p2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader()
        w.writerows(summ)
    print(f"[OK] {p2}")
    for s in summ:
        if s["portion"] == "ALL":
            print(f"  {s['model']:8s} {s['clusterer']:8s} {s['regime']:6s} "
                  f"obs={s['mean_f1_observed']:.4f} "
                  f"null={s['mean_f1_null']:.4f} ({s['null_kind']}) "
                  f"single={s['mean_f1_single_cluster']:.4f}")


if __name__ == "__main__":
    main()
