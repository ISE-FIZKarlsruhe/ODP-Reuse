#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seed_sensitivity.py -- how much of a per-ontology RQ5 number is the random seed?

WHY
---
Both embedders set ``seed=42`` and ``workers=1``, and both clusterers are seeded,
so the pipeline looks deterministic. It is not. ``gensim.models.Word2Vec``
defaults to ``hashfxn=hash``; Python's string ``hash`` is salted per process
unless ``PYTHONHASHSEED`` is fixed, and gensim uses it to seed each word's
initial vector. Verified behaviour: two runs in the SAME process are bit
identical; two runs in different processes with the same ``PYTHONHASHSEED`` are
bit identical; two runs with different ``PYTHONHASHSEED`` are not. The published
run did not fix it.

This script quantifies the consequence by comparing ``cluster_diagnostics.py``
dumps produced under different ``PYTHONHASHSEED`` values over the same
ontologies: how often the selected configuration changes, how far the
per-ontology F1 moves, and how far the aggregate moves.

USAGE
-----
    python seed_sensitivity.py \
        --dump 0=".../diag_seed0.jsonl.shard1" \
        --dump 1=".../diag_seed1.jsonl.shard1" \
        --dump 2=".../diag_seed2.jsonl.shard1" \
        --out ../results/iri/clusterstats
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
from pathlib import Path

import numpy as np


def load(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            out += [json.loads(l) for l in fh if l.strip()]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", action="append", required=True,
                   help="SEED=glob, repeatable")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dumps: dict[str, list[dict]] = {}
    for spec in args.dump:
        seed, pat = spec.split("=", 1)
        dumps[seed] = load(pat)
        print(f"seed {seed}: {len(dumps[seed])} rows")

    # index selected rows per (regime, ontology, method, clusterer)
    sel: dict[str, dict] = {}
    for seed, rows in dumps.items():
        d = {}
        for r in rows:
            for reg, flag in (("oracle", "is_oracle"), ("unsup", "is_unsup")):
                if r[flag]:
                    d[(reg, r["dataset"], r["ontology"], r["method"],
                       r["clusterer"])] = r
        sel[seed] = d

    seeds = sorted(dumps)
    common_onts = set.intersection(*[{k[1:3] for k in sel[s]} for s in seeds])
    print(f"{len(common_onts)} ontologies common to all seeds")

    rows_out: list[dict] = []
    for a, b in itertools.combinations(seeds, 2):
        for reg in ("oracle", "unsup"):
            for m in ("owl2vec", "rdf2vec"):
                for cl in ("kmeans", "hdbscan"):
                    keys = [k for k in sel[a]
                            if k[0] == reg and k[3] == m and k[4] == cl
                            and k[1:3] in common_onts and k in sel[b]]
                    if not keys:
                        continue
                    ra = [sel[a][k] for k in keys]
                    rb = [sel[b][k] for k in keys]
                    same_cfg = np.mean([x["config"] == y["config"]
                                        for x, y in zip(ra, rb)])
                    df1 = np.array([x["f1"] - y["f1"] for x, y in zip(ra, rb)])
                    dnp = np.array([x["n_predicted"] - y["n_predicted"]
                                    for x, y in zip(ra, rb)], dtype=float)
                    rows_out.append({
                        "seed_a": a, "seed_b": b, "regime": reg, "model": m,
                        "clusterer": cl, "n_ontologies": len(keys),
                        "pct_same_selected_config": round(100 * float(same_cfg), 2),
                        "mean_f1_a": round(float(np.mean([x["f1"] for x in ra])), 4),
                        "mean_f1_b": round(float(np.mean([y["f1"] for y in rb])), 4),
                        "aggregate_f1_delta": round(
                            float(np.mean([x["f1"] for x in ra])
                                  - np.mean([y["f1"] for y in rb])), 4),
                        "mean_abs_per_ontology_f1_delta":
                            round(float(np.mean(np.abs(df1))), 4),
                        "median_abs_per_ontology_f1_delta":
                            round(float(np.median(np.abs(df1))), 4),
                        "p90_abs_per_ontology_f1_delta":
                            round(float(np.percentile(np.abs(df1), 90)), 4),
                        "pct_ontologies_f1_delta_gt_0_10":
                            round(100 * float(np.mean(np.abs(df1) > 0.10)), 2),
                        "mean_abs_n_predicted_delta":
                            round(float(np.mean(np.abs(dnp))), 3),
                        "max_abs_n_predicted_delta": int(np.abs(dnp).max()),
                    })

    tag = f"_{args.tag}" if args.tag else ""
    p = args.out / f"t7_seed_sensitivity{tag}.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    print(f"[OK] {p}  ({len(rows_out)} rows)")


if __name__ == "__main__":
    main()
