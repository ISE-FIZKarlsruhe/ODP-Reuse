#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rq5_recommendation.py -- RQ5: what does the ODP-Reuse dataset buy an existing
ODP recommender?

The question is not "which recommender is best" but "does having a corpus of
confirmed ontology-to-pattern reuse improve ODP recommendation over the prior
state of the art?". ODPReco (Doc2Vec similarity against a fixed 73-ODP catalogue)
is the prior system, and it uses no reuse corpus at all: it cannot learn that
ontologies about agents tend to reuse `agentrole`, because it has no examples.

Three tiers are therefore compared, and the improvement is measured as a PAIRED
per-ontology difference so the comparison is not confounded by which ontologies
happen to be easy:

  TIER 0  no reuse data          random ranking; content retrieval against
                                 ODPReco's fixed 73-ODP catalogue
  TIER 1  ODPReco's approach     Doc2Vec content matching (a faithful
                                 re-implementation -- see the honesty note)
  TIER 2  ODP-Reuse informed     popularity prior over the reuse matrix;
                                 ontology-to-ontology collaborative filtering

HONESTY NOTE -- read before citing
----------------------------------
No results from the original ODPReco jar exist in this release. The TIER 1 row is
a *correct gensim re-implementation* of ODPReco's Doc2Vec scoring idea, trained
properly on the whole corpus; ODPReco itself re-fits Doc2Vec on one-line corpora
per run, so it is expected to be weaker, not stronger. Write "a faithful
re-implementation of ODPReco's content-matching approach", never "ODPReco scores
X". The catalogue-level comparison (TIER 0's fixed catalogue vs a dataset-derived
one) uses ODPReco's actual shipped 73-ODP catalogue.

Inputs come from the recommendation use case, which holds the per-ontology
results needed for paired tests:

    use-cases/odp-recommendation/results/<dataset>/*.json

Usage
-----
    python run_rq5_recommendation.py --use-case ../use-cases/odp-recommendation \\
        --dataset noOBO --out out/rq5_reco
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BS = "\\"
EOL = BS + BS

# (tier, backend file stem, display label, uses the ODP-Reuse dataset?)
TIERS = [
    (0, "random", "Random ranking", "no"),
    (0, "embed_item_FIXEDcatalog[st_all-MiniLM-L6-v2]",
     "Content retrieval, ODPReco's fixed 73-ODP catalogue", "no"),
    (1, "doc2vec", "Doc2Vec content matching (ODPReco's approach)$^{\\dagger}$",
     "no"),
    (2, "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]",
     "Content retrieval, ODP-Reuse-derived catalogue", "yes"),
    (2, "popularity", "Popularity prior over the reuse matrix", "yes"),
    (2, "embed_ontology_knn_cf",
     "Collaborative filtering (embedding similarity)", "yes"),
    (2, "ontology_knn_cf", "Collaborative filtering (Jaccard similarity)", "yes"),
]

# The improvement claims RQ5 makes. (label, treatment, control, note)
IMPROVEMENTS = [
    ("ODP-Reuse-informed CF vs ODPReco's approach",
     "ontology_knn_cf", "doc2vec",
     "the headline improvement: what the reuse corpus adds over content matching"),
    ("ODP-Reuse-informed CF vs no reuse data at all",
     "ontology_knn_cf", "random",
     "the floor: is any of this better than chance"),
    ("ODP-Reuse-derived catalogue vs ODPReco's fixed catalogue",
     "embed_item_datasetcatalog[st_all-MiniLM-L6-v2]",
     "embed_item_FIXEDcatalog[st_all-MiniLM-L6-v2]",
     "isolates the catalogue, holding the embedding model constant"),
    ("Popularity prior vs ODPReco's approach",
     "popularity", "doc2vec",
     "even the simplest use of the reuse data beats content matching"),
]

METRICS = [("map", "MAP"), ("precision@1", "P@1"), ("recall@10", "R@10"),
           ("mrr", "MRR"), ("ndcg@10", "nDCG@10")]


def load_per(results_dir: Path, stem: str) -> list[dict] | None:
    p = results_dir / f"{stem}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("per_ontology", d.get("evaluation"))


def load_summary(results_dir: Path, stem: str) -> dict:
    p = results_dir / f"{stem}.json"
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8")).get("summary", {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--use-case", required=True, type=Path,
                    help="path to use-cases/odp-recommendation")
    ap.add_argument("--dataset", default="noOBO")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.use_case))
    from recobench.stats import bootstrap_ci, paired_delta  # noqa: E402

    results = args.use_case / "results" / args.dataset
    if not results.is_dir():
        sys.exit(f"[ERROR] no results at {results}")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---------------- per-recommender table
    rows = []
    for tier, stem, label, uses in TIERS:
        per = load_per(results, stem)
        if per is None:
            print(f"[SKIP] {stem}: not present")
            continue
        summary = load_summary(results, stem)
        ci = bootstrap_ci(per, "map", n_boot=args.n_boot, seed=args.seed)
        row = {"tier": tier, "recommender": stem, "label": label,
               "uses_odp_reuse_data": uses, "n_ontologies": ci["n"],
               "map": ci["mean"], "map_ci_lo": ci["lo"], "map_ci_hi": ci["hi"]}
        for key, _lab in METRICS:
            if key == "map":
                continue
            v = summary.get(f"mean_{key}")
            if v is None and per:
                v = sum(float(r.get(key, 0) or 0) for r in per) / len(per)
            row[key] = round(float(v), 4) if v is not None else ""
        rows.append(row)
    rows.sort(key=lambda r: (r["tier"], -r["map"]))
    write_csv(args.out / f"rq5_recommenders_{args.dataset}.csv", rows)

    # ---------------- paired improvements
    imps = []
    for label, treat, ctrl, note in IMPROVEMENTS:
        a, b = load_per(results, treat), load_per(results, ctrl)
        if a is None or b is None:
            print(f"[SKIP] {label}: missing backend")
            continue
        d = paired_delta(a, b, "map", n_boot=args.n_boot, seed=args.seed)
        if not d.get("n"):
            continue
        sig = (d["ci_lo"] > 0 or d["ci_hi"] < 0) and d["p_sign_test"] < 0.05
        base = next((r["map"] for r in rows if r["recommender"] == ctrl), None)
        imps.append({
            "improvement": label, "treatment": treat, "control": ctrl,
            "n_paired": d["n"], "delta_map": d["mean_delta"],
            "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
            "relative_pct": (round(100.0 * d["mean_delta"] / base, 1)
                             if base else ""),
            "p_bootstrap": d["p_bootstrap"], "p_sign_test": d["p_sign_test"],
            "win_loss_tie": f"{d['wins']}/{d['losses']}/{d['ties']}",
            "significant_at_0.05": "yes" if sig else "no",
            "note": note,
        })
    write_csv(args.out / f"rq5_improvements_{args.dataset}.csv", imps)

    # ---------------- LaTeX
    tex = args.out / "tables"
    tex.mkdir(exist_ok=True)
    n = rows[0]["n_ontologies"] if rows else 0
    tier_name = {0: "no reuse data", 1: "ODPReco's approach",
                 2: "ODP-Reuse informed"}

    L = [BS + "begin{table}[t]", BS + "centering", BS + "small",
         BS + r"caption{RQ5: benefit of the ODP-Reuse dataset for ODP "
         r"recommendation. MAP with a $95" + BS + r"%$ percentile-bootstrap "
         r"confidence interval over the " + str(n) + r" held-out ontologies. "
         r"Recommenders are grouped by whether they can consult a corpus of "
         r"confirmed reuse. $^{" + BS + r"dagger}$A faithful re-implementation of "
         r"ODPReco's content-matching approach, trained properly on the whole "
         r"corpus; the original re-fits Doc2Vec per run on one-line corpora and is "
         r"expected to be weaker.}",
         BS + r"label{tab:rq5_reco}",
         BS + r"setlength{" + BS + r"tabcolsep}{4pt}",
         BS + r"begin{tabular}{ll|ccc}", BS + "toprule",
         " & ".join([BS + r"textbf{Uses reuse data}", BS + r"textbf{Recommender}",
                     BS + r"textbf{MAP}", BS + r"textbf{95" + BS + r"% CI}",
                     BS + r"textbf{P@1}"]) + " " + EOL,
         BS + "midrule"]
    prev = None
    for r in rows:
        if r["tier"] != prev:
            if prev is not None:
                L.append(BS + "midrule")
            tag = BS + r"emph{" + tier_name[r["tier"]] + "}"
            prev = r["tier"]
        else:
            tag = ""
        best = r["recommender"] == "ontology_knn_cf"
        m = (BS + "textbf{" + f"{r['map']:.3f}" + "}") if best else f"{r['map']:.3f}"
        lab = (BS + "textbf{" + r["label"] + "}") if best else r["label"]
        L.append(" & ".join([tag, lab, m,
                             f"$[{r['map_ci_lo']:.3f}, {r['map_ci_hi']:.3f}]$",
                             f"{r.get('precision@1', 0):.3f}"]) + " " + EOL)
    L += [BS + "bottomrule", BS + r"end{tabular}", BS + r"end{table}"]
    (tex / "rq5_recommenders.tex").write_text("\n".join(L) + "\n", encoding="utf-8")

    # improvements table
    L2 = [BS + "begin{table}[t]", BS + "centering", BS + "small",
          BS + r"caption{RQ5: paired improvement in MAP from using the ODP-Reuse "
          r"dataset. Each row compares two recommenders on the SAME held-out "
          r"ontologies, so the difference is not confounded by ontology "
          r"difficulty. CI is a $95" + BS + r"%$ bootstrap interval on the mean "
          r"per-ontology difference; $p$ is an exact two-sided sign test; W/L/T "
          r"counts ontologies won, lost and tied.}",
          BS + r"label{tab:rq5_improvement}",
          BS + r"setlength{" + BS + r"tabcolsep}{4pt}",
          BS + r"begin{tabular}{l|rrlr}", BS + "toprule",
          " & ".join([BS + r"textbf{Comparison}", BS + r"textbf{$" + BS +
                      r"Delta$ MAP}", BS + r"textbf{95" + BS + r"% CI}",
                      BS + r"textbf{W/L/T}", BS + r"textbf{$p$}"]) + " " + EOL,
          BS + "midrule"]
    for im in imps:
        star = BS + "textbf{" if im["significant_at_0.05"] == "yes" else ""
        end = "}" if star else ""
        L2.append(" & ".join([
            im["improvement"],
            f"{star}{im['delta_map']:+.3f}{end}",
            f"$[{im['ci_lo']:+.3f}, {im['ci_hi']:+.3f}]$",
            im["win_loss_tie"],
            ("$<0.001$" if im["p_sign_test"] < 0.001
             else f"{im['p_sign_test']:.3f}"),
        ]) + " " + EOL)
    L2 += [BS + "bottomrule", BS + r"end{tabular}", BS + r"end{table}"]
    (tex / "rq5_improvement.tex").write_text("\n".join(L2) + "\n", encoding="utf-8")

    print(f"\n[OK] tables in {tex}")
    print("\nRQ5 improvements (bold = significant at 0.05):")
    for im in imps:
        flag = "SIGNIFICANT" if im["significant_at_0.05"] == "yes" else "not sig."
        print(f"  {im['improvement']}")
        print(f"      dMAP {im['delta_map']:+.3f}  CI [{im['ci_lo']:+.3f}, "
              f"{im['ci_hi']:+.3f}]  W/L/T {im['win_loss_tie']}  "
              f"p={im['p_sign_test']:.3f}  -> {flag}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"[WARN] no rows for {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {path.name} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
