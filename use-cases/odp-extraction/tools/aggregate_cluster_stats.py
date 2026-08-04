#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_cluster_stats.py -- turn the raw dumps into the reviewer-facing tables.

INPUTS
------
--published-jsonl   results/<track>/rq5_per_ontology.jsonl, i.e. the run behind
                    the paper's table. It stores the SELECTED config name and
                    P/R/F1 but not the cluster count. For K-Means the count is
                    recoverable exactly (config ``kmeans_k9`` -> 9 clusters, an
                    identity verified against the re-run in
                    ``pred_count_identity_check.csv``); for HDBSCAN it is not
                    recoverable at all, because only ``min_cluster_size`` is
                    recorded.
--diagnostics       cluster_diagnostics.py shard files (glob), which DO carry the
                    counts, sizes, noise and split/merge structure -- from a
                    re-run with PYTHONHASHSEED fixed, not from the published
                    process. See the caveat table.
--gt-overlap        gt_overlap_per_ontology.csv from gt_overlap_stats.py.

OUTPUTS (all under --out)
-------------------------
t1_pred_vs_gt_counts.csv        predicted vs ground-truth pattern counts
t1b_pred_gt_diff_bins.csv       distribution of (predicted - ground truth)
t2_cluster_size_coverage.csv    cluster sizes, entity coverage, noise share
t3_split_merge.csv              split / merge / unmatched-cluster confusion
t5_matching_artefact.csv        effect of averaging over matched pairs only
t6_selected_config_shift.csv    published vs re-run selected-config agreement
pred_count_identity_check.csv   is n_predicted == k for K-Means?  (per config)

Every table is emitted per (portion, model, clusterer, regime); ``regime`` is
``oracle`` (hyper-parameters chosen by argmax F1 against the ground truth, i.e.
an upper bound) or ``unsup`` (chosen by silhouette). Under ORACLE selection k was
tuned against the ground truth, so any agreement between predicted and
ground-truth counts there is partly circular and is labelled as such.

USAGE
-----
    python aggregate_cluster_stats.py \
        --published-jsonl ../results/iri/rq5_per_ontology.jsonl \
        --diagnostics "../results/iri/clusterstats/raw/diag_seed0.jsonl.shard*" \
        --gt-overlap ../results/iri/clusterstats/gt_overlap_per_ontology.csv \
        --out ../results/iri/clusterstats
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

K_RE = re.compile(r"^kmeans_k(\d+)$")
REGIMES = ("oracle", "unsup")
MODELS = ("owl2vec", "rdf2vec")
CLUSTERERS = ("kmeans", "hdbscan")


def _q(a, p):
    return round(float(np.percentile(a, p)), 3) if len(a) else ""


def _m(a):
    return round(float(np.mean(a)), 3) if len(a) else ""


def _md(a):
    return round(float(np.median(a)), 3) if len(a) else ""


def load_published(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def load_diag(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no diagnostics files matched {pattern!r}")
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    print(f"loaded {len(out)} diagnostic rows from {len(files)} shard(s)")
    return out


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[WARN] nothing to write for {path.name}")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"[OK] {path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
# T1: predicted vs ground-truth counts
# ----------------------------------------------------------------------

def count_stats(portion, model, clusterer, regime, source, pred, gt) -> dict:
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    diff = pred - gt
    ratio = pred / gt
    rho = pv = ""
    if len(pred) > 2 and np.std(pred) > 0 and np.std(gt) > 0:
        r = spearmanr(pred, gt)
        rho, pv = round(float(r.statistic), 3), float(f"{r.pvalue:.3g}")
    return {
        "portion": portion, "model": model, "clusterer": clusterer,
        "regime": regime, "source": source, "n_ontologies": len(pred),
        "mean_pred": _m(pred), "median_pred": _md(pred),
        "sd_pred": round(float(np.std(pred, ddof=1)), 3) if len(pred) > 1 else "",
        "p25_pred": _q(pred, 25), "p75_pred": _q(pred, 75),
        "mean_gt": _m(gt), "median_gt": _md(gt),
        "p25_gt": _q(gt, 25), "p75_gt": _q(gt, 75),
        "ratio_of_means": round(float(np.mean(pred) / np.mean(gt)), 3),
        "median_of_per_ontology_ratio": _md(ratio),
        "mean_diff": _m(diff), "median_diff": _md(diff),
        "p10_diff": _q(diff, 10), "p90_diff": _q(diff, 90),
        "min_diff": int(diff.min()), "max_diff": int(diff.max()),
        "pct_exact": round(100 * float(np.mean(diff == 0)), 2),
        "pct_within_1": round(100 * float(np.mean(np.abs(diff) <= 1)), 2),
        "pct_under": round(100 * float(np.mean(diff < 0)), 2),
        "pct_over": round(100 * float(np.mean(diff > 0)), 2),
        "spearman_pred_gt": rho, "spearman_p": pv,
        # The K-Means grid is k = 2..10, so an ontology reusing a single pattern
        # CANNOT be matched exactly by construction. Separating the ontologies
        # where an exact match is expressible avoids reading a structural
        # impossibility as a modelling failure.
        "pct_gt_eq_1": round(100 * float(np.mean(gt == 1)), 2),
        "pct_gt_in_k_grid_2_10": round(100 * float(np.mean((gt >= 2) & (gt <= 10))), 2),
        "pct_exact_where_gt_in_k_grid": round(
            100 * float(np.mean(diff[(gt >= 2) & (gt <= 10)] == 0)), 2)
        if np.any((gt >= 2) & (gt <= 10)) else "",
    }


def diff_bins(portion, model, clusterer, regime, source, pred, gt) -> dict:
    d = np.asarray(pred, dtype=float) - np.asarray(gt, dtype=float)
    n = len(d)
    row = {"portion": portion, "model": model, "clusterer": clusterer,
           "regime": regime, "source": source, "n_ontologies": n}
    bins = [("le_-3", d <= -3), ("-2_to_-1", (d >= -2) & (d <= -1)),
            ("0", d == 0), ("+1_to_+2", (d >= 1) & (d <= 2)),
            ("+3_to_+5", (d >= 3) & (d <= 5)), ("+6_to_+9", (d >= 6) & (d <= 9)),
            ("ge_+10", d >= 10)]
    for name, mask in bins:
        row[f"n_{name}"] = int(mask.sum())
        row[f"pct_{name}"] = round(100 * float(mask.mean()), 2)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--published-jsonl", required=True, type=Path)
    ap.add_argument("--diagnostics", default="")
    ap.add_argument("--gt-overlap", type=Path, default=None)
    ap.add_argument("--pairs-csv", type=Path, default=None,
                    help="pipeline/out/<rq>/pairs.csv, for the count-consistency check")
    ap.add_argument("--manifest-csv", type=Path, default=None,
                    help="pipeline/out/MANIFEST.csv, for the count-consistency check")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    pub = load_published(args.published_jsonl)
    diag = load_diag(args.diagnostics) if args.diagnostics else []
    portions = sorted({r["dataset"] for r in pub})

    # ----------------------------------------------------------- T0
    # Does the ground-truth count the extraction run scored against agree with
    # the count the RQ statistics and the manifest report?  If these three ever
    # disagree, every table below describes a different ground truth.
    t0: list[dict] = []
    jn = {(r["dataset"], r["ontology"]): r["n_gt"] for r in pub}
    pairs_n: dict[tuple[str, str], int] = defaultdict(int)
    if args.pairs_csv:
        for r in csv.DictReader(args.pairs_csv.open(encoding="utf-8")):
            if (r.get("gate_passed") == "1" and not r.get("drop_reason")
                    and int(r["n_reused"]) > 0):
                pairs_n[(r["dataset"], r["ontology"])] += 1
    man_n: dict[tuple[str, str], int] = {}
    if args.manifest_csv:
        for r in csv.DictReader(args.manifest_csv.open(encoding="utf-8")):
            man_n[(r["dataset"], r["ontology_id"])] = int(r["n_gt_patterns"] or 0)
    if pairs_n or man_n or diag:
        dn = {(r["dataset"], r["ontology"]): r["n_gt"] for r in diag}
        for name, other in (("pairs.csv", pairs_n), ("MANIFEST.n_gt_patterns", man_n),
                            ("cluster_diagnostics_rerun", dn)):
            if not other:
                continue
            t0.append({
                "reference": name, "n_ontologies_in_published_jsonl": len(jn),
                "n_present_in_reference": sum(1 for k in jn if k in other),
                "n_agree_on_n_gt": sum(1 for k, v in jn.items() if other.get(k) == v),
                "n_disagree": sum(1 for k, v in jn.items()
                                  if k in other and other[k] != v),
            })
    # Sizes, not just counts: gt_overlap_stats.py re-derives the ground truth
    # from the corpus, so its instance sizes must equal pairs.csv's n_reused
    # multiset per ontology. Equal counts with unequal sizes would mean the two
    # derivations disagree about WHICH entities are reused.
    if args.gt_overlap and args.gt_overlap.is_file() and args.pairs_csv:
        psz: dict[tuple[str, str], list[int]] = defaultdict(list)
        for r in csv.DictReader(args.pairs_csv.open(encoding="utf-8")):
            if (r.get("gate_passed") == "1" and not r.get("drop_reason")
                    and int(r["n_reused"]) > 0):
                psz[(r["dataset"], r["ontology"])].append(int(r["n_reused"]))
        same = diff = 0
        for r in csv.DictReader(args.gt_overlap.open(encoding="utf-8")):
            mine = sorted(int(x) for x in r["gt_sizes"].split(";"))
            if sorted(psz.get((r["dataset"], r["ontology"]), [])) == mine:
                same += 1
            else:
                diff += 1
        t0.append({"reference": "pairs.csv n_reused (instance SIZES)",
                   "n_ontologies_in_published_jsonl": same + diff,
                   "n_agree_on_n_gt": same, "n_disagree": diff})

    # ontologies for which no clustering configuration existed at all
    noconf = sorted({(r["dataset"], r["ontology"], r["n_entities"]) for r in pub
                     if not r["oracle_config"]})
    for ds, oid, ne in noconf:
        t0.append({"reference": "no_candidate_configuration",
                   "ontology": f"{ds}/{oid}", "n_entities": ne,
                   "note": "k>=n for every k in the grid; scored 0 by construction"})
    write(args.out / f"t0_gt_count_consistency{tag}.csv", t0)

    # --------------------------------------------------------------- T1
    t1: list[dict] = []
    t1b: list[dict] = []
    # (a) published run, K-Means only: n_predicted == k, recoverable from config
    for ds in portions + ["ALL"]:
        for m in MODELS:
            for reg in REGIMES:
                sub = [r for r in pub
                       if (ds == "ALL" or r["dataset"] == ds)
                       and r["method"] == m and r["clusterer"] == "kmeans"
                       and K_RE.match(r[f"{reg}_config"] or "")]
                if not sub:
                    continue
                pred = [int(K_RE.match(r[f"{reg}_config"]).group(1)) for r in sub]
                gt = [r["n_gt"] for r in sub]
                t1.append(count_stats(ds, m, "kmeans", reg, "published", pred, gt))
                t1b.append(diff_bins(ds, m, "kmeans", reg, "published", pred, gt))
    # (b) re-run, both clusterers, from the stored n_predicted
    if diag:
        for ds in portions + ["ALL"]:
            for m in MODELS:
                for cl in CLUSTERERS:
                    for reg in REGIMES:
                        key = "is_oracle" if reg == "oracle" else "is_unsup"
                        sub = [r for r in diag
                               if (ds == "ALL" or r["dataset"] == ds)
                               and r["method"] == m and r["clusterer"] == cl
                               and r[key]]
                        if not sub:
                            continue
                        pred = [r["n_predicted"] for r in sub]
                        gt = [r["n_gt"] for r in sub]
                        t1.append(count_stats(ds, m, cl, reg, "rerun_seed0", pred, gt))
                        t1b.append(diff_bins(ds, m, cl, reg, "rerun_seed0", pred, gt))
    write(args.out / f"t1_pred_vs_gt_counts{tag}.csv", t1)
    write(args.out / f"t1b_pred_gt_diff_bins{tag}.csv", t1b)

    # the per-ontology record the two T1 tables summarise, so the distribution
    # can be re-plotted or re-tested without re-running anything
    tidy: list[dict] = []
    for r in pub:
        for reg in REGIMES:
            mo = K_RE.match(r[f"{reg}_config"] or "")
            if r["clusterer"] == "kmeans" and mo:
                tidy.append({"source": "published", "portion": r["dataset"],
                             "ontology": r["ontology"], "model": r["method"],
                             "clusterer": "kmeans", "regime": reg,
                             "config": r[f"{reg}_config"],
                             "n_entities": r["n_entities"], "n_gt": r["n_gt"],
                             "n_predicted": int(mo.group(1)),
                             "diff": int(mo.group(1)) - r["n_gt"]})
    for r in diag:
        for reg, flag in (("oracle", "is_oracle"), ("unsup", "is_unsup")):
            if r[flag]:
                tidy.append({"source": "rerun_seed0", "portion": r["dataset"],
                             "ontology": r["ontology"], "model": r["method"],
                             "clusterer": r["clusterer"], "regime": reg,
                             "config": r["config"], "n_entities": r["n_entities"],
                             "n_gt": r["n_gt"], "n_predicted": r["n_predicted"],
                             "diff": r["n_predicted"] - r["n_gt"]})
    write(args.out / f"per_ontology_pred_vs_gt{tag}.csv", tidy)

    if not diag:
        print("no diagnostics supplied: T2/T3/T5/T6 skipped")
        return

    # --------------------------------------------------------------- T2/T3/T5
    t2: list[dict] = []
    t3: list[dict] = []
    t5: list[dict] = []
    for ds in portions + ["ALL"]:
        for m in MODELS:
            for cl in CLUSTERERS:
                for reg in REGIMES:
                    key = "is_oracle" if reg == "oracle" else "is_unsup"
                    sub = [r for r in diag
                           if (ds == "ALL" or r["dataset"] == ds)
                           and r["method"] == m and r["clusterer"] == cl and r[key]]
                    if not sub:
                        continue
                    base = {"portion": ds, "model": m, "clusterer": cl,
                            "regime": reg, "n_ontologies": len(sub)}

                    sizes = [s for r in sub for s in r["cluster_sizes"]]
                    gtsizes = [s for r in sub for s in r["gt_sizes"]]
                    largest_share = [max(r["cluster_sizes"]) / r["n_entities"]
                                     for r in sub if r["cluster_sizes"]]
                    t2.append(dict(base, **{
                        "n_clusters_total": len(sizes),
                        "mean_cluster_size": _m(sizes),
                        "median_cluster_size": _md(sizes),
                        "p90_cluster_size": _q(sizes, 90),
                        "max_cluster_size": int(max(sizes)) if sizes else "",
                        "pct_clusters_of_size_2": round(
                            100 * float(np.mean(np.asarray(sizes) == 2)), 2) if sizes else "",
                        "mean_gt_instance_size": _m(gtsizes),
                        "median_gt_instance_size": _md(gtsizes),
                        "size_ratio_median_cluster_over_median_gt": round(
                            float(np.median(sizes) / np.median(gtsizes)), 3)
                        if sizes and gtsizes else "",
                        "mean_entity_coverage": _m([r["coverage"] for r in sub]),
                        "median_entity_coverage": _md([r["coverage"] for r in sub]),
                        "mean_noise_share": round(
                            1 - float(np.mean([r["coverage"] for r in sub])), 4),
                        "mean_largest_cluster_share_of_ontology": _m(largest_share),
                        "mean_gt_frac_in_noise": _m(
                            [x for r in sub for x in r["gt_frac_in_noise"]]),
                        "pct_gt_instances_fully_in_noise": round(100 * float(np.mean(
                            [x == 1.0 for r in sub for x in r["gt_frac_in_noise"]])), 2),
                    }))

                    hits = [h for r in sub for h in r["gt_n_clusters_hit"]]
                    hits2 = [h for r in sub for h in r["gt_n_clusters_hit2"]]
                    merges = [h for r in sub for h in r["pred_n_gt_hit"]]
                    t3.append(dict(base, **{
                        "n_gt_instances": len(hits),
                        "mean_clusters_per_gt_instance": _m(hits),
                        "median_clusters_per_gt_instance": _md(hits),
                        "max_clusters_per_gt_instance": int(max(hits)) if hits else "",
                        "pct_gt_instances_split_ge2_clusters": round(
                            100 * float(np.mean(np.asarray(hits) >= 2)), 2) if hits else "",
                        "pct_gt_instances_split_ge3_clusters": round(
                            100 * float(np.mean(np.asarray(hits) >= 3)), 2) if hits else "",
                        "pct_gt_instances_split_ge2_clusters_min2entities": round(
                            100 * float(np.mean(np.asarray(hits2) >= 2)), 2) if hits2 else "",
                        "pct_gt_instances_in_1_cluster": round(
                            100 * float(np.mean(np.asarray(hits) == 1)), 2) if hits else "",
                        "pct_gt_instances_in_0_clusters": round(
                            100 * float(np.mean(np.asarray(hits) == 0)), 2) if hits else "",
                        "n_clusters": len(merges),
                        "pct_clusters_touching_0_gt": round(
                            100 * float(np.mean(np.asarray(merges) == 0)), 2) if merges else "",
                        "pct_clusters_touching_1_gt": round(
                            100 * float(np.mean(np.asarray(merges) == 1)), 2) if merges else "",
                        "pct_clusters_merging_ge2_gt": round(
                            100 * float(np.mean(np.asarray(merges) >= 2)), 2) if merges else "",
                        "mean_gt_instances_per_cluster": _m(merges),
                    }))

                    # T5: the average runs over min(n_pred, n_gt) matched pairs
                    mm = np.asarray([min(r["n_predicted"], r["n_gt"]) for r in sub],
                                    dtype=float)
                    ngt = np.asarray([r["n_gt"] for r in sub], dtype=float)
                    f1 = np.asarray([r["f1"] for r in sub], dtype=float)
                    rc = np.asarray([r["recall"] for r in sub], dtype=float)
                    pr = np.asarray([r["precision"] for r in sub], dtype=float)
                    npred = np.asarray([r["n_predicted"] for r in sub], dtype=float)
                    safe_npred = np.where(npred > 0, npred, 1.0)
                    t5.append(dict(base, **{
                        "pct_ontologies_npred_lt_ngt": round(
                            100 * float(np.mean(npred < ngt)), 2),
                        "n_ontologies_zero_clusters": int((npred == 0).sum()),
                        "n_gt_instances_never_matched": int((ngt - mm).sum()),
                        "pct_gt_instances_never_matched": round(
                            100 * float((ngt - mm).sum() / ngt.sum()), 2),
                        "f1_as_published": round(float(np.mean(f1)), 4),
                        "f1_all_gt_instances": round(float(np.mean(f1 * mm / ngt)), 4),
                        "recall_as_published": round(float(np.mean(rc)), 4),
                        "recall_all_gt_instances": round(float(np.mean(rc * mm / ngt)), 4),
                        "precision_as_published": round(float(np.mean(pr)), 4),
                        "precision_all_clusters": round(
                            float(np.mean(pr * mm / safe_npred)), 4),
                    }))
    write(args.out / f"t2_cluster_size_coverage{tag}.csv", t2)
    write(args.out / f"t3_split_merge{tag}.csv", t3)
    write(args.out / f"t5_matching_artefact{tag}.csv", t5)

    # --------------------------------------------------------------- T4
    # Are predicted clusters disjoint?  ``_labels_to_clusters`` gives each entity
    # exactly one label, so they must be -- but the claim is checkable rather
    # than assumed: the diagnostics store cluster sizes and the noise count
    # computed from the SET UNION of the clusters, so
    #   sum(cluster_sizes) + n_noise == n_entities
    # holds if and only if no entity appears in two clusters.
    t4: list[dict] = []
    for cl in CLUSTERERS:
        sub = [r for r in diag if r["clusterer"] == cl]
        ok = sum(1 for r in sub
                 if sum(r["cluster_sizes"]) + r["n_noise"] == r["n_entities"])
        t4.append({
            "clusterer": cl, "n_clusterings_checked": len(sub),
            "n_clusterings_forming_a_partition": ok,
            "n_clusterings_with_overlapping_clusters": len(sub) - ok,
            "n_clusterings_with_any_noise": sum(1 for r in sub if r["n_noise"] > 0),
            "verdict": ("hard partition of the non-noise entities"
                        if ok == len(sub) else "OVERLAP DETECTED"),
        })
    write(args.out / f"t4_partition_check{tag}.csv", t4)

    # --------------------------------------------------------------- identity check
    idc: list[dict] = []
    km = [r for r in diag if r["clusterer"] == "kmeans"]
    for cfg in sorted({r["config"] for r in km}, key=lambda c: int(K_RE.match(c).group(1))):
        sub = [r for r in km if r["config"] == cfg]
        k = int(K_RE.match(cfg).group(1))
        idc.append({"config": cfg, "k": k, "n_rows": len(sub),
                    "n_rows_where_n_predicted_eq_k":
                        sum(1 for r in sub if r["n_predicted"] == k),
                    "n_rows_where_n_predicted_lt_k":
                        sum(1 for r in sub if r["n_predicted"] < k),
                    "min_n_predicted": min(r["n_predicted"] for r in sub),
                    "max_n_predicted": max(r["n_predicted"] for r in sub)})
    write(args.out / f"pred_count_identity_check{tag}.csv", idc)

    # --------------------------------------------------------------- T9
    # "Would knowing the right number of patterns help?"  For every ontology whose
    # ground-truth count falls inside the K-Means grid, compare the configuration
    # k = n_gt against the two selected configurations. If cardinality were the
    # binding constraint, k = n_gt would dominate; if entity assignment is the
    # binding constraint, it will not.
    t9: list[dict] = []
    for ds in portions + ["ALL"]:
        for m in MODELS:
            groups: dict[tuple, list[dict]] = defaultdict(list)
            for r in diag:
                if (r["clusterer"] == "kmeans" and r["method"] == m
                        and (ds == "ALL" or r["dataset"] == ds)
                        and 2 <= r["n_gt"] <= 10):
                    groups[(r["dataset"], r["ontology"])].append(r)
            recs = []
            for grid in groups.values():
                want = f"kmeans_k{grid[0]['n_gt']}"
                hit = [r for r in grid if r["config"] == want]
                if not hit:
                    continue          # k >= n_entities, config not in the grid
                orc = [r for r in grid if r["is_oracle"]][0]
                uns = [r for r in grid if r["is_unsup"]][0]
                recs.append((hit[0], orc, uns))
            if not recs:
                continue
            t9.append({
                "portion": ds, "model": m, "clusterer": "kmeans",
                "n_ontologies_with_2_le_ngt_le_10": len(recs),
                "mean_f1_k_eq_ngt": round(
                    float(np.mean([a["f1"] for a, _b, _c in recs])), 4),
                "mean_f1_oracle_selected": round(
                    float(np.mean([b["f1"] for _a, b, _c in recs])), 4),
                "mean_f1_unsup_selected": round(
                    float(np.mean([c["f1"] for _a, _b, c in recs])), 4),
                "pct_ontologies_where_k_eq_ngt_is_the_oracle_choice": round(
                    100 * float(np.mean([a["config"] == b["config"]
                                         for a, b, _c in recs])), 2),
                "pct_ontologies_where_k_eq_ngt_beats_unsup": round(
                    100 * float(np.mean([a["f1"] > c["f1"]
                                         for a, _b, c in recs])), 2),
                "mean_oracle_k": round(
                    float(np.mean([int(K_RE.match(b["config"]).group(1))
                                   for _a, b, _c in recs])), 2),
                "mean_ngt": round(
                    float(np.mean([a["n_gt"] for a, _b, _c in recs])), 2),
            })
    write(args.out / f"t9_k_equals_ngt{tag}.csv", t9)

    # --------------------------------------------------------------- T6
    pubmap = {(r["dataset"], r["ontology"], r["method"], r["clusterer"]): r
              for r in pub}
    t6: list[dict] = []
    for ds in portions + ["ALL"]:
        for m in MODELS:
            for cl in CLUSTERERS:
                for reg in REGIMES:
                    key = "is_oracle" if reg == "oracle" else "is_unsup"
                    sub = [r for r in diag
                           if (ds == "ALL" or r["dataset"] == ds)
                           and r["method"] == m and r["clusterer"] == cl and r[key]]
                    pairs = [(r, pubmap.get((r["dataset"], r["ontology"],
                                             r["method"], r["clusterer"])))
                             for r in sub]
                    pairs = [(a, b) for a, b in pairs if b]
                    if not pairs:
                        continue
                    same = sum(1 for a, b in pairs
                               if a["config"] == b[f"{reg}_config"])
                    df1 = [a["f1"] - b[f"{reg}_f1"] for a, b in pairs]
                    t6.append({"portion": ds, "model": m, "clusterer": cl,
                               "regime": reg, "n_ontologies": len(pairs),
                               "pct_same_selected_config":
                                   round(100 * same / len(pairs), 2),
                               "mean_f1_published":
                                   round(float(np.mean([b[f"{reg}_f1"] for _a, b in pairs])), 4),
                               "mean_f1_rerun":
                                   round(float(np.mean([a["f1"] for a, _b in pairs])), 4),
                               "mean_abs_per_ontology_f1_delta":
                                   round(float(np.mean(np.abs(df1))), 4),
                               "median_abs_per_ontology_f1_delta":
                                   round(float(np.median(np.abs(df1))), 4),
                               "p90_abs_per_ontology_f1_delta":
                                   round(float(np.percentile(np.abs(df1), 90)), 4),
                               "pct_ontologies_f1_delta_gt_0_10":
                                   round(100 * float(np.mean(np.abs(df1) > 0.10)), 2)})
    write(args.out / f"t6_selected_config_shift{tag}.csv", t6)


if __name__ == "__main__":
    main()
