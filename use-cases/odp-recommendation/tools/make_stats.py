#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_stats.py -- regenerate the tidy statistics CSVs under ``stats/``.

Everything here is DERIVED from the stored per-ontology result files in
``results/<dataset>/*.json``; nothing is transcribed by hand from a report.
Re-running this script after a new benchmark run refreshes every number the
paper cites.

Outputs (for --dataset noOBO):

  stats/reco_headline.csv        one row per recommender: MAP/P@1/R@10/nDCG@10,
                                 whether it uses the ODP-Reuse data, n, and a
                                 ``citable`` flag (see CAVEATS.md).
  stats/reco_ci_map.csv          percentile-bootstrap 95% CI of mean MAP per
                                 recommender (5000 resamples, seed 0).
  stats/reco_paired_tests.csv    paired per-ontology deltas for the comparisons
                                 the paper makes, each with a bootstrap CI, a
                                 bootstrap p-value and an exact sign-test p.
  stats/catalog_coverage.csv     ODPReco fixed-catalog coverage of the dataset's
                                 pattern vocabulary (the recall-ceiling table).

Usage
-----
    python tools/make_stats.py --dataset noOBO
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from recobench.stats import bootstrap_ci, paired_delta  # noqa: E402

# Comparisons the paper makes.  Each is (label, backend_a, backend_b, note).
PAIRED_COMPARISONS = [
    ("dataset-informed CF vs no-dataset floor",
     "ontology_knn_cf", "random",
     "headline claim: does the ODP-Reuse data help at all"),
    ("dataset catalogue vs ODPReco fixed catalogue (same encoder)",
     "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]",
     "embed_item_FIXEDcatalog[st_all-MiniLM-L6-v2]",
     "isolates the catalogue, holding the embedding model constant"),
    ("CF vs popularity prior",
     "ontology_knn_cf", "popularity",
     "is CF better than 'recommend the most reused patterns'"),
    ("CF vs Doc2Vec content matching",
     "ontology_knn_cf", "doc2vec",
     "CF vs a correct re-implementation of ODPReco's scoring idea"),
    ("LLM rerank vs its retrieval base",
     "llm_rerank[qwen2.5_7b-instruct]", "embed_ontology_knn_cf",
     "n is small (LLM sample); directional only"),
]

# Which rows may be cited as evidence, and why not when they may not.
# See CAVEATS.md -- this is the citation gate.
CITABILITY = {
    "random": ("YES", "floor baseline"),
    "popularity": ("YES", "full test set, n=23"),
    "ontology_knn_cf": ("YES", "full test set, n=23"),
    "embed_ontology_knn_cf": ("YES", "full test set, n=23"),
    "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]": ("YES", "full test set, n=23"),
    "embed_item_FIXEDcatalog[st_all-MiniLM-L6-v2]": ("YES", "full test set, n=23"),
    "doc2vec": ("QUALIFIED",
                "re-implementation of ODPReco's Doc2Vec idea, NOT the ODPReco jar; "
                "label it as such"),
    "llm_rerank[qwen2.5_7b-instruct]": ("QUALIFIED", "n=8 sample; directional only"),
    "llm_generative[qwen2.5_7b-instruct]": ("QUALIFIED", "n=8 sample; directional only"),
}

HEADLINE_METRICS = [
    ("map", "MAP"),
    ("precision@1", "P@1"),
    ("recall@5", "R@5"),
    ("recall@10", "R@10"),
    ("mrr", "MRR"),
    ("ndcg@10", "nDCG@10"),
    ("hits@10", "Hit@10"),
]


def backend_name(path: Path) -> str:
    return path.stem


def load_per(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("per_ontology", data.get("evaluation", []))


def load_summary(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", {})


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[OK] {path.relative_to(HERE)}  ({len(rows)} rows)")


def uses_dataset(name: str) -> str:
    if name == "random":
        return "no"
    if "FIXEDcatalog" in name:
        return "no (ODPReco's fixed 73-ODP catalogue)"
    return "yes"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="noOBO")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=HERE / "stats")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results_dir = args.results_dir or (HERE / "results" / args.dataset)
    if not results_dir.is_dir():
        sys.exit(f"[ERROR] no results directory: {results_dir}")

    files = sorted(results_dir.glob("*.json"))
    if not files:
        sys.exit(f"[ERROR] no result JSONs in {results_dir}")

    per_by_backend = {backend_name(f): load_per(f) for f in files}
    summary_by_backend = {backend_name(f): load_summary(f) for f in files}

    # ---------------------------------------------------------------- headline
    rows = []
    for name, per in per_by_backend.items():
        summary = summary_by_backend.get(name, {})
        citable, note = CITABILITY.get(name, ("REVIEW", "not classified"))
        row = [name, uses_dataset(name), len(per)]
        for key, _label in HEADLINE_METRICS:
            value = summary.get(f"mean_{key}")
            if value is None and per:
                value = sum(float(r.get(key, 0.0) or 0.0) for r in per) / len(per)
            row.append(round(float(value), 4) if value is not None else "")
        row += [citable, note]
        rows.append(row)
    rows.sort(key=lambda r: (r[3] if r[3] != "" else -1), reverse=True)
    write_csv(
        args.out_dir / f"reco_headline_{args.dataset}.csv",
        ["recommender", "uses_odp_reuse_data", "n_test_ontologies"]
        + [label for _k, label in HEADLINE_METRICS]
        + ["citable", "citation_note"],
        rows,
    )

    # ------------------------------------------------------------- bootstrap CI
    ci_rows = []
    for name, per in per_by_backend.items():
        ci = bootstrap_ci(per, "map", n_boot=args.n_boot, seed=args.seed)
        citable, _note = CITABILITY.get(name, ("REVIEW", ""))
        ci_rows.append([name, ci["n"], ci["mean"], ci["lo"], ci["hi"],
                        round(ci["hi"] - ci["lo"], 4), citable])
    ci_rows.sort(key=lambda r: r[2], reverse=True)
    write_csv(
        args.out_dir / f"reco_ci_map_{args.dataset}.csv",
        ["recommender", "n", "mean_map", "ci_lo_95", "ci_hi_95", "ci_width", "citable"],
        ci_rows,
    )

    # ------------------------------------------------------------ paired tests
    paired_rows = []
    for label, a, b, note in PAIRED_COMPARISONS:
        if a not in per_by_backend or b not in per_by_backend:
            print(f"[SKIP] {label}: missing backend ({a} or {b})")
            continue
        d = paired_delta(per_by_backend[a], per_by_backend[b], "map",
                         n_boot=args.n_boot, seed=args.seed)
        if not d.get("n"):
            print(f"[SKIP] {label}: no common ontologies")
            continue
        # A difference is called significant only if the 95% CI excludes 0
        # AND the sign test agrees at 0.05.
        significant = (d["ci_lo"] > 0 or d["ci_hi"] < 0) and d["p_sign_test"] < 0.05
        paired_rows.append([
            label, a, b, d["n"], d["mean_delta"], d["ci_lo"], d["ci_hi"],
            d["p_bootstrap"], d["p_sign_test"],
            f"{d['wins']}/{d['losses']}/{d['ties']}",
            "yes" if significant else "no", note,
        ])
    write_csv(
        args.out_dir / f"reco_paired_tests_{args.dataset}.csv",
        ["comparison", "method_a", "method_b", "n_paired", "mean_delta_map",
         "ci_lo_95", "ci_hi_95", "p_bootstrap", "p_sign_test", "win_loss_tie",
         "significant_at_0.05", "note"],
        paired_rows,
    )

    # -------------------------------------------------------- catalog coverage
    map_file = HERE / "pattern_catalog_map.json"
    if map_file.is_file():
        cat = json.loads(map_file.read_text(encoding="utf-8"))
        mapped = {k: v for k, v in cat.get("mapped", {}).items()
                  if not k.startswith("_")}
        unmapped = {k: v for k, v in cat.get("unmapped", {}).items()
                    if not k.startswith("_")}
        cov_rows = [[stem, "mapped", target] for stem, target in sorted(mapped.items())]
        cov_rows += [[stem, "unmapped", ""] for stem in sorted(unmapped)]
        write_csv(
            args.out_dir / "catalog_coverage.csv",
            ["gt_pattern_stem", "status", "odpreco_catalog_odp"],
            cov_rows,
        )
        total = len(mapped) + len(unmapped)
        print(f"       fixed catalogue size = {cat.get('_catalog_size')}; "
              f"{len(mapped)}/{total} observed GT stems reachable "
              f"({100 * len(mapped) / total:.0f}%)")


if __name__ == "__main__":
    main()
