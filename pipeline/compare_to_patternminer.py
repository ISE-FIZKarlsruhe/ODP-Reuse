#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_to_patternminer.py -- reconcile this pipeline against PatternMiner's
`aggregates/`, on both ontology COUNTS and evaluation RESULTS.

Both papers describe the same corpus, so their numbers must reconcile or the
difference must be explained. This script does the reconciliation mechanically:

  1. POPULATION -- who evaluated which ontologies, and for every ontology that
     PatternMiner evaluated but this pipeline drops, WHICH GATE RULE dropped it.
     That turns "PatternMiner has more ontologies" into an itemised explanation.

  2. RESULTS -- mean F1 for OWL2Vec* and RDF2Vec, compared three ways:
       - each side on its own population (not comparable, shown for context)
       - both sides restricted to the COMMON ontologies (comparable)
       - this pipeline's oracle vs unsupervised regimes against PatternMiner,
         whose selection is oracle by construction

Outputs `comparison_to_patternminer.{csv,md}`.

Usage
-----
    python compare_to_patternminer.py --manifest out/MANIFEST.csv \\
        --aggregates <PatternMiner>/aggregates \\
        --extraction ../use-cases/odp-extraction/results/gated-reuse \\
        --out out/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

PM_FILES = {"github": "per_ontology_f1_github.csv",
            "noOBO": "per_ontology_f1_noOBO.csv"}
METHODS = ("owl2vec", "rdf2vec")


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--aggregates", required=True, type=Path)
    ap.add_argument("--extraction", required=True, type=Path,
                    help="dir holding extraction_per_ontology.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = {(r["dataset"], r["ontology_id"]): r
                for r in csv.DictReader(args.manifest.open(encoding="utf-8"))}
    mine = [json.loads(l) for l in
            (args.extraction / "extraction_per_ontology.jsonl").open(encoding="utf-8")]

    pop_rows, res_rows, expl_rows = [], [], []

    for ds, fname in PM_FILES.items():
        pmf = args.aggregates / fname
        if not pmf.is_file():
            print(f"[SKIP] {ds}: {pmf} missing")
            continue
        pm = list(csv.DictReader(pmf.open(encoding="utf-8")))
        pm_onto = {r["ontology"] for r in pm}
        my_onto = {r["ontology"] for r in mine if r["dataset"] == ds}
        common = my_onto & pm_onto

        # ---- 1. population reconciliation
        only_pm = pm_onto - my_onto
        reasons = Counter()
        for o in only_pm:
            r = manifest.get((ds, o))
            if r is None:
                reasons["not_in_manifest"] += 1
            elif r["drop_reason"]:
                reasons[r["drop_reason"].split(":", 1)[0]] += 1
            else:
                reasons["kept_but_not_evaluated"] += 1
        pop_rows.append({
            "dataset": ds,
            "this_pipeline": len(my_onto),
            "patternminer": len(pm_onto),
            "common": len(common),
            "only_this_pipeline": len(my_onto - pm_onto),
            "only_patternminer": len(only_pm),
        })
        for k, v in reasons.most_common():
            expl_rows.append({"dataset": ds, "gate_rule_that_drops_it": k,
                              "n_ontologies": v,
                              "pct_of_only_patternminer": round(
                                  100.0 * v / max(len(only_pm), 1), 1)})

        # ---- 2. results
        for m in METHODS:
            pm_all = [float(r["macro_f1"]) for r in pm
                      if r["method"] == m and r["macro_f1"] not in ("", "nan")]
            pm_com = [float(r["macro_f1"]) for r in pm
                      if r["method"] == m and r["ontology"] in common
                      and r["macro_f1"] not in ("", "nan")]
            my_km = [r for r in mine if r["dataset"] == ds
                     and r["method"] == m and r["clusterer"] == "kmeans"]
            my_all_o = [r["oracle_f1"] for r in my_km]
            my_all_u = [r["unsup_f1"] for r in my_km]
            my_com_o = [r["oracle_f1"] for r in my_km if r["ontology"] in common]
            my_com_u = [r["unsup_f1"] for r in my_km if r["ontology"] in common]
            res_rows.append({
                "dataset": ds, "method": m,
                "pm_n": len(pm_all), "pm_mean_f1": round(mean(pm_all), 4),
                "this_n": len(my_all_o),
                "this_oracle_f1": round(mean(my_all_o), 4),
                "this_unsup_f1": round(mean(my_all_u), 4),
                "common_n": len(pm_com),
                "pm_f1_common": round(mean(pm_com), 4),
                "this_oracle_f1_common": round(mean(my_com_o), 4),
                "this_unsup_f1_common": round(mean(my_com_u), 4),
                "delta_oracle_vs_pm_common": round(
                    mean(my_com_o) - mean(pm_com), 4),
            })

    def dump(name, rows):
        if not rows:
            return
        p = args.out / name
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {p.name}")

    dump("population_reconciliation.csv", pop_rows)
    dump("population_explanation.csv", expl_rows)
    dump("results_comparison.csv", res_rows)

    md = ["# Reconciliation with PatternMiner `aggregates/`", "",
          "## 1. Population", "",
          "| dataset | this pipeline | PatternMiner | common | only here | only PM |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in pop_rows:
        md.append(f"| {r['dataset']} | {r['this_pipeline']} | {r['patternminer']} "
                  f"| {r['common']} | {r['only_this_pipeline']} | "
                  f"{r['only_patternminer']} |")
    md += ["", "## 2. Why PatternMiner has more ontologies", "",
           "Each row is a gate rule this pipeline applies and PatternMiner does not.",
           "",
           "| dataset | dropped by | n | % of PM-only |", "|---|---|---:|---:|"]
    for r in expl_rows:
        md.append(f"| {r['dataset']} | {r['gate_rule_that_drops_it']} | "
                  f"{r['n_ontologies']} | {r['pct_of_only_patternminer']}% |")
    md += ["", "## 3. Results (K-Means)", "",
           "`pm` = PatternMiner, whose hyper-parameters are selected against the",
           "ground truth, so it is comparable to the **oracle** column only.", "",
           "| dataset | method | PM n | PM F1 | this n | oracle | unsup | common n | PM F1 (common) | oracle (common) | unsup (common) | Δ oracle−PM |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res_rows:
        md.append(f"| {r['dataset']} | {r['method']} | {r['pm_n']} | "
                  f"{r['pm_mean_f1']} | {r['this_n']} | {r['this_oracle_f1']} | "
                  f"{r['this_unsup_f1']} | {r['common_n']} | {r['pm_f1_common']} | "
                  f"{r['this_oracle_f1_common']} | {r['this_unsup_f1_common']} | "
                  f"{r['delta_oracle_vs_pm_common']} |")
    (args.out / "comparison_to_patternminer.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] comparison_to_patternminer.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
