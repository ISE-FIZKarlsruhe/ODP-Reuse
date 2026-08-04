#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_tracks.py -- put the exact-IRI and iri+label tracks side by side.

The two tracks differ in ONE thing: whether human-adjudicated label matches
extend the ground truth. Everything else -- corpus, gate rules, thresholds,
seeds -- is identical. The difference between them is therefore a measurement of
what exact-IRI matching misses, which is the question the previous paper's
Limitations section calls unanswerable.

Reads whatever exists and reports the rest as pending, so it can be run while
jobs are still finishing.

Usage
    python compare_tracks.py --out out/comparison_tracks
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

TRACKS = {
    "iri": {
        "label": "exact IRI",
        "manifest": HERE / "out" / "MANIFEST.csv",
        "gate": HERE / "out" / "gate_summary.csv",
        "rq_gated": HERE / "out" / "rq_iri_gated",
        "rq_all": HERE / "out" / "rq_iri_all",
        "uc2": REPO / "use-cases/odp-extraction/results/iri/rq5_results.csv",
        "uc1": REPO / "use-cases/odp-recommendation/results_reco/iri",
    },
    "iri-label": {
        "label": "iri + adjudicated label",
        "manifest": HERE / "out_iri_label" / "MANIFEST.csv",
        "gate": HERE / "out_iri_label" / "gate_summary.csv",
        "rq_gated": HERE / "out" / "rq_iri-label_gated",
        "rq_all": HERE / "out" / "rq_iri-label_all",
        "uc2": REPO / "use-cases/odp-extraction/results/iri-label/rq5_results.csv",
        "uc1": REPO / "use-cases/odp-recommendation/results_reco/iri-label",
    },
}
PORTION = {"github": "GitHub", "noOBO": "Paper &amp; Repo", "OBO": "DOS-DP"}


def read(p: Path) -> list[dict]:
    if not p or not Path(p).is_file():
        return []
    with Path(p).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def fnum(v, d=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return d


def delta(a, b, fmt="{:+d}"):
    if a is None or b is None:
        return "—"
    return fmt.format(b - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "out" / "comparison_tracks")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    L: list[str] = ["# Exact-IRI vs iri+label", "",
                    "The tracks differ in exactly one respect: whether the 71 "
                    "human-adjudicated label matches extend the ground truth. "
                    "Corpus, gate rules, thresholds and seeds are identical, so "
                    "every difference below is attributable to that choice.", ""]

    # ---------------------------------------------------------------- gate
    L += ["## 1. Quality gate", "",
          "| Portion | kept (iri) | kept (iri+label) | Δ | complete (iri) | "
          "complete (iri+label) | Δ |", "|---|---:|---:|---:|---:|---:|---:|"]
    gates = {k: {r["dataset"]: r for r in read(v["gate"])}
             for k, v in TRACKS.items()}
    tot = {"a": 0, "b": 0, "ca": 0, "cb": 0}
    for ds in ("github", "noOBO", "OBO"):
        a, b = gates["iri"].get(ds), gates["iri-label"].get(ds)
        if not a or not b:
            continue
        ka, kb = num(a["kept_reuse"]), num(b["kept_reuse"])
        ca, cb = num(a["kept_complete_reuse"]), num(b["kept_complete_reuse"])
        tot["a"] += ka; tot["b"] += kb; tot["ca"] += ca; tot["cb"] += cb
        L.append(f"| {PORTION.get(ds, ds)} | {ka} | {kb} | {kb - ka:+d} | "
                 f"{ca} | {cb} | {cb - ca:+d} |")
    if tot["a"]:
        L.append(f"| **total** | **{tot['a']}** | **{tot['b']}** | "
                 f"**{tot['b'] - tot['a']:+d}** | **{tot['ca']}** | "
                 f"**{tot['cb']}** | **{tot['cb'] - tot['ca']:+d}** |")
    L.append("")

    # what the labels actually added
    man = read(TRACKS["iri-label"]["manifest"])
    ext = [r for r in man if num(r.get("n_label_confirmed")) > 0]
    if ext:
        L += [f"Label extension: **{len(ext)} ontologies**, "
              f"**{sum(num(r['n_label_confirmed']) for r in ext)} patterns**, "
              f"**{sum(num(r['n_label_entities']) for r in ext)} entities** "
              f"added to the ground truth.", "",
              "Complete-reuse is unchanged: re-modelled reuse is partial by "
              "nature, so it extends a pattern's coverage without completing "
              "it.", ""]

    # ---------------------------------------------------------------- RQ
    for pop, key in (("gated", "rq_gated"), ("real corpus", "rq_all")):
        ra = {r["dataset"]: r for r in read(Path(TRACKS["iri"][key]) / "rq1_extent.csv")}
        rb = {r["dataset"]: r for r in read(Path(TRACKS["iri-label"][key]) / "rq1_extent.csv")}
        if not ra and not rb:
            L += [f"## RQ1 extent ({pop})", "", "_pending_", ""]
            continue
        L += [f"## RQ1 extent ({pop})", "",
              "| Portion | onto (iri) | onto (i+l) | Δ | patterns (iri) | "
              "patterns (i+l) | mean p/o (iri) | mean p/o (i+l) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for ds in ("github", "noOBO", "OBO"):
            a, b = ra.get(ds), rb.get(ds)
            if not a and not b:
                continue
            L.append(
                f"| {PORTION.get(ds, ds)} "
                f"| {num(a.get('onto_reuse')) if a else '—'} "
                f"| {num(b.get('onto_reuse')) if b else '—'} "
                f"| {delta(num(a.get('onto_reuse')) if a else None, num(b.get('onto_reuse')) if b else None)} "
                f"| {num(a.get('patterns_reuse')) if a else '—'} "
                f"| {num(b.get('patterns_reuse')) if b else '—'} "
                f"| {fnum(a.get('mean_patterns_per_onto')):.2f} " if a else "| — "
                f"| {fnum(b.get('mean_patterns_per_onto')):.2f} |" if b else "| — |")
        L.append("")

    # ---------------------------------------------------------------- UC2
    ua, ub = read(TRACKS["iri"]["uc2"]), read(TRACKS["iri-label"]["uc2"])
    L += ["## UC2 · automatic extraction", ""]
    if ua or ub:
        ka = {(r["dataset"], r["method"], r["clusterer"]): r for r in ua}
        kb = {(r["dataset"], r["method"], r["clusterer"]): r for r in ub}
        L += ["| Portion | Model | Clust. | oracle F₁ (iri) | oracle F₁ (i+l) | "
              "unsup F₁ (iri) | unsup F₁ (i+l) |",
              "|---|---|---|---:|---:|---:|---:|"]
        for k in sorted(set(ka) | set(kb)):
            a, b = ka.get(k), kb.get(k)
            L.append(f"| {PORTION.get(k[0], k[0])} | "
                     f"{'OWL2Vec*' if k[1] == 'owl2vec' else 'RDF2Vec'} | "
                     f"{'K-Means' if k[2] == 'kmeans' else 'HDBSCAN'} | "
                     f"{fnum(a['oracle_f1']):.3f} | " if a else "| — | "
                     f"{fnum(b['oracle_f1']):.3f} | " if b else "— | "
                     f"{fnum(a['unsup_f1']):.3f} | " if a else "— | "
                     f"{fnum(b['unsup_f1']):.3f} |" if b else "— |")
        L.append("")
    else:
        L += ["_pending_", ""]

    # ---------------------------------------------------------------- UC1
    L += ["## UC1 · recommendation", ""]
    any_uc1 = False
    for portion in ("noOBO", "github"):
        fa = Path(TRACKS["iri"]["uc1"]) / portion / "comparison.csv"
        fb = Path(TRACKS["iri-label"]["uc1"]) / portion / "comparison.csv"
        ra, rb = read(fa), read(fb)
        if not ra or not rb:
            continue
        any_uc1 = True
        A = {r["recommender"]: r for r in ra}
        B = {r["recommender"]: r for r in rb}
        n_a = num(ra[0].get("n_ontologies"))
        gt_a, gt_b = fnum(ra[0].get("mean_n_truth")), fnum(rb[0].get("mean_n_truth"))
        L += [f"### {PORTION.get(portion, portion)} (n={n_a} test ontologies, "
              f"mean GT patterns {gt_a:.2f} → {gt_b:.2f})", "",
              "| Recommender | MAP (iri) | MAP (i+l) | Δ |",
              "|---|---:|---:|---:|"]
        for k in A:
            if k in B:
                x, y = fnum(A[k]["mean_map"]), fnum(B[k]["mean_map"])
                L.append(f"| {k[:46]} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
        L.append("")
    if not any_uc1:
        L += ["_pending_", ""]

    # ---------------------------------------------------------------- reading
    L += ["## How to read this", "",
          "The label track is **not** a better dataset; it is a different "
          "question. Exact IRI answers *who copied a pattern verbatim*, with "
          "perfect precision by construction. iri+label additionally answers "
          "*who re-implemented one*, at the cost of human adjudication — "
          "automated label matching alone has precision 0.215 (95% CI "
          "[0.174, 0.262]), so it cannot be harvested unattended.", "",
          "Report both. The delta is the measurement.", ""]

    p = args.out / "COMPARISON.md"
    p.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[OK] {p}")


if __name__ == "__main__":
    main()
