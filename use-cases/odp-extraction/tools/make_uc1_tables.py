#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_uc1_tables.py -- one UC1 table over every method, on both corpora.

Reads the per-ontology records of the two runners:

    results/<dir>/methods_per_ontology.jsonl   run_method_eval.py
    results/<dir>/rq5_per_ontology.jsonl       run_extraction_eval.py

and writes a single LaTeX table plus the tidy CSV behind it.

PAIRED POPULATIONS
------------------
The IRI-anonymised corpus is missing the ontologies whose hashing failed, so a
plain score over 442 ontologies and an anonymised score over 424 are not
comparable. Every plain-vs-anonymised number here is computed over the
INTERSECTION of the two populations, per dataset, and the intersection size is
reported. Scores over the full plain population are written to the CSV as well,
so the effect of the restriction is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DS_LABEL = {"noOBO": "Paper \\& Repo", "OBO": "DOS--DP", "github": "GitHub"}

ROWS = [
    ("group", "Reference points", None),
    ("all_in_one", "All entities in one cluster", None),
    ("singletons", "One entity per cluster", None),
    ("connected_components", "Connected components", None),
    ("namespace", "IRI namespace grouping", None),
    ("group", "Published extraction methods", None),
    ("asprino_cc", "Conceptual components, communities~\cite{10.1145/3460210.3493542_extraction}", None),
    ("asprino_cc_full", "Conceptual components, full pipeline~\cite{10.1145/3460210.3493542_extraction}", None),
    ("law_ap", "Axiom-tree mining, axiom patterns~\cite{doi:10.3233/SW-170280_extraction}", None),
    ("law_cfp", "Axiom-tree mining, class frames~\cite{doi:10.3233/SW-170280_extraction}", None),
    ("group", "Embedding baselines", None),
    ("owl2vec|kmeans", "OWL2Vec*~\cite{chen2021owl2vec} + K-Means", ("owl2vec", "kmeans")),
    ("owl2vec|hdbscan", "OWL2Vec* + HDBSCAN", ("owl2vec", "hdbscan")),
    ("rdf2vec|kmeans", "RDF2Vec~\cite{ristoski2016rdf2vec} + K-Means", ("rdf2vec", "kmeans")),
    ("rdf2vec|hdbscan", "RDF2Vec + HDBSCAN", ("rdf2vec", "hdbscan")),
]


def load(run_dir: Path) -> dict[tuple[str, str, str], dict]:
    """(dataset, ontology, method) -> {precision, recall, f1, n_predicted}."""
    out: dict[tuple[str, str, str], dict] = {}
    for name in ("methods_per_ontology.jsonl",
                 "asprino_full_per_ontology.jsonl"):
        mp = run_dir / name
        if not mp.is_file():
            continue
        for line in mp.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["dataset"], r["ontology"], r["method"])
            # A method's population comes from one run: the full-pipeline file
            # re-emits its own step 1, which the methods run already supplied
            # under a different cost bound. First writer wins.
            out.setdefault(key, r)
    ep = run_dir / "rq5_per_ontology.jsonl"
    if ep.is_file():
        # run_extraction_eval.py does not record the cluster count;
        # aggregate_cluster_stats.py does, in the unsupervised regime.
        npred: dict[tuple[str, str, str], float] = {}
        cs = run_dir / "clusterstats" / "per_ontology_pred_vs_gt.csv"
        if cs.is_file():
            for r in csv.DictReader(cs.open(encoding="utf-8")):
                if r.get("regime") != "unsup":
                    continue
                npred[(r["portion"], r["ontology"],
                       f'{r["model"]}|{r["clusterer"]}')] = float(r["n_predicted"])
        for line in ep.open(encoding="utf-8"):
            r = json.loads(line)
            key = (r["dataset"], r["ontology"], f'{r["method"]}|{r["clusterer"]}')
            out[key] = {"precision": r["unsup_precision"],
                        "recall": r["unsup_recall"], "f1": r["unsup_f1"],
                        "n_predicted": npred.get(key, float("nan")),
                        "oracle_f1": r["oracle_f1"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plain", nargs="+", required=True, type=Path,
                    help="one or more run directories for the plain corpus")
    ap.add_argument("--anon", nargs="*", default=[], type=Path,
                    help="one or more run directories for the anonymised corpus")
    ap.add_argument("--datasets", nargs="*", default=["github", "noOBO"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plain: dict = {}
    for d in args.plain:
        plain.update(load(d))
    anon: dict = {}
    for d in args.anon:
        anon.update(load(d))

    methods = [m for m, _lab, _e in ROWS if m != "group"]
    rows: list[dict] = []
    pair_n: dict[str, int] = {}

    # One paired population per dataset: the ontologies scored by EVERY method
    # in BOTH corpora. A per-method intersection would give each row its own
    # denominator and make the rows incomparable to each other.
    # The paired population is the set of ontologies every method covers on
    # both corpora. The full conceptual-components pipeline is excluded from
    # that intersection: it only produces output where step 1 produced a
    # community, so requiring it would shrink the population for every other
    # method too. Its own n is reported in its row instead.
    POPULATION_EXEMPT = {"asprino_cc_full"}
    common: dict[str, list[str]] = {}
    for ds in args.datasets:
        sets = []
        for m in methods:
            if m in POPULATION_EXEMPT:
                continue
            for src in (plain, anon):
                if not src:
                    continue
                s = {o for (d, o, mm) in src if d == ds and mm == m}
                if s:
                    sets.append(s)
        common[ds] = sorted(set.intersection(*sets)) if sets else []
        pair_n[ds] = len(common[ds])

    for ds in args.datasets:
        for m in methods:
            p_keys = {o for (d, o, mm) in plain if d == ds and mm == m}
            a_keys = {o for (d, o, mm) in anon if d == ds and mm == m}
            if m in POPULATION_EXEMPT:
                both = sorted(p_keys)          # plain only; see the caption
            else:
                both = [o for o in common[ds] if o in p_keys and o in a_keys]
            row = {"dataset": ds, "method": m,
                   "n_plain_full": len(p_keys), "n_paired": len(both)}
            for tag, src, keys in (("plain_full", plain, sorted(p_keys)),
                                   ("plain", plain, both),
                                   ("anon", anon, both)):
                # A method exempt from the paired population has no anonymised
                # counterpart, and a run can be missing an ontology another run
                # has. Score only over what this source actually holds.
                keys = [o for o in keys if (ds, o, m) in src]
                if not keys:
                    continue
                vals = [src[(ds, o, m)] for o in keys]
                for k in ("precision", "recall", "f1"):
                    row[f"{tag}_{k}"] = round(
                        float(np.mean([v[k] for v in vals])), 4)
                npred = [v["n_predicted"] for v in vals
                         if v.get("n_predicted") == v.get("n_predicted")]
                if npred:
                    row[f"{tag}_n_predicted"] = round(float(np.mean(npred)), 2)
            if "plain_f1" in row and "anon_f1" in row:
                row["delta_f1_anon"] = round(row["anon_f1"] - row["plain_f1"], 4)
            rows.append(row)

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (args.out / "uc1_all_methods.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {args.out / 'uc1_all_methods.csv'}")

    idx = {(r["dataset"], r["method"]): r for r in rows}
    # Bold the best METHOD, not the best row: the reference points are not
    # methods, and marking a degenerate one "best" would read as a result.
    REFERENCE = {"all_in_one", "singletons", "connected_components", "namespace"}
    real = [m for m in methods if m not in REFERENCE]
    best = {ds: max((idx[(ds, m)].get("plain_f1", -1) for m in real
                     if (ds, m) in idx), default=-1) for ds in args.datasets}
    best_anon = {ds: max((idx[(ds, m)].get("anon_f1", -1) for m in real
                          if (ds, m) in idx), default=-1) for ds in args.datasets}

    BS, EOL = "\\", "\\\\"
    L: list[str] = [
        BS + "begin{table*}[t]", BS + "centering",
        BS + "caption{UC1: automatic ODP extraction on the quality-gated corpus. "
        r"Entity-level precision, recall and $F_1$, macro-averaged over "
        r"ontologies after Hungarian matching of predicted clusters to "
        r"ground-truth pattern instances. \emph{pred.} is the mean number of "
        r"clusters predicted per ontology. Every score is computed over the "
        r"ontologies present in \emph{both} corpora, so the plain and "
        r"anonymised columns are paired; population sizes are given per "
        r"dataset. The full conceptual-components pipeline is the exception: "
        r"it needs the lexical content of entity labels for its "
        r"disambiguation and frame-detection steps, which IRI anonymisation "
        r"removes, so it is reported on the plain corpus only and over its "
        r"own population. Embedding hyper-parameters are chosen by silhouette without "
        r"consulting the ground truth. The best $F_1$ per column among the "
        r"\emph{methods} is in bold; the reference points at the top are not "
        r"methods and are excluded from that comparison. Clustering is by K-Means~\cite{lloyd1982kmeans} or HDBSCAN~\cite{campello2013hdbscan}, and matching by Hungarian assignment~\cite{kuhn1955hungarian}.}",
        BS + r"label{tab:uc1_all_methods}",
        BS + r"setlength{" + BS + r"tabcolsep}{4pt}", BS + "footnotesize",
        BS + r"begin{tabular}{l|rrrr|r|rrrr|r}", BS + "toprule",
        r"& \multicolumn{5}{c|}{\textbf{GitHub}} & "
        r"\multicolumn{5}{c}{\textbf{Paper \& Repo}} " + EOL,
        r"& \multicolumn{4}{c|}{plain} & anon. & "
        r"\multicolumn{4}{c|}{plain} & anon. " + EOL,
        r"\textbf{Method} & pred. & P & R & $F_1$ & $F_1$ "
        r"& pred. & P & R & $F_1$ & $F_1$ " + EOL,
        BS + "midrule",
    ]
    for m, lab, _e in ROWS:
        if m == "group":
            L.append(r"\multicolumn{11}{l}{\textit{" + lab + "}} " + EOL)
            continue
        cells = [lab]
        for ds in args.datasets:
            r = idx.get((ds, m))
            if not r or "plain_f1" not in r:
                cells += ["--"] * 5
                continue
            f1 = r["plain_f1"]
            af1 = r.get("anon_f1")
            cells += [
                f"{r.get('plain_n_predicted', float('nan')):.2f}"
                if r.get("plain_n_predicted") is not None else "--",
                f"{r['plain_precision']:.3f}", f"{r['plain_recall']:.3f}",
                # A reference point can tie the best method -- connected
                # components does on GitHub -- and must still not be bolded.
                (r"\textbf{" + f"{f1:.3f}" + "}")
                if m not in REFERENCE and abs(f1 - best[ds]) < 1e-9
                else f"{f1:.3f}",
                "--" if af1 is None else
                ((r"\textbf{" + f"{af1:.3f}" + "}")
                 if m not in REFERENCE and abs(af1 - best_anon[ds]) < 1e-9
                 else f"{af1:.3f}"),
            ]
        L.append(" & ".join(cells) + " " + EOL)
    npair = " ".join(f"{DS_LABEL.get(ds, ds)} $n={pair_n.get(ds, 0)}$;"
                     for ds in args.datasets)
    # Why the singleton reference point scores P<1, which is the first thing a
    # reader checks. Fixed text: these are properties of the frozen v1.2.0
    # ground truth, not of a run. Recompute with
    #   tools/gt_overlap_stats.py --from-csv \
    #       results/iri/clusterstats/gt_overlap_per_ontology.csv
    # (n_pairs_overlapping > 0 and partition_ceiling_f1 give every figure).
    CEILING_NOTE = (
        r"\emph{One entity per cluster} scores $P=0.992$ rather than $1.000$ "
        r"because each ground-truth instance is matched to at most one "
        r"predicted cluster: where an ontology's instances share entities, "
        r"only one instance can claim the singleton holding a shared entity, "
        r"and the remaining instances are matched to singletons lying outside "
        r"them. Ten GitHub ontologies are affected. The constraint binds every "
        r"method and not this reference point alone --- 73 of the 442 GitHub "
        r"and 12 of the 51 Paper \& Repo ontologies have instances that "
        r"overlap, and on those no hard partition can reach $F_1$ 1. The exact "
        r"bound averages $0.964$ and $0.976$ over the two portions and falls "
        r"to $0.333$ at worst."
    )
    # The one row not on the shared population, and the like-for-like figure a
    # reader needs in order to compare it with the row above it. Recompute with
    #   results/asprino-full/asprino_full_per_ontology.jsonl, which emits both
    #   asprino_cc and asprino_cc_full over its own single population.
    POPULATION_NOTE = (
        r"\emph{Populations.} Every method is scored over the same ontologies, "
        r"the intersection of those both corpora provide, with one exception: "
        r"steps 2--7 of the full conceptual-components pipeline consume the "
        r"communities its first stage produces, and on 35 GitHub and 2 Paper "
        r"\& Repo ontologies that stage produced none, so the pipeline is "
        r"reported over the 407 and 49 it covers. Scored over those same "
        r"ontologies its first stage reaches $F_1$ 0.286 and 0.330."
    )
    L += [BS + "bottomrule", BS + r"end{tabular}",
          BS + r"begin{flushleft}\footnotesize Paired populations: " + npair
          + EOL + "\n" + CEILING_NOTE + EOL + "\n" + POPULATION_NOTE
          + r"\end{flushleft}",
          BS + r"end{table*}"]
    (args.out / "uc1_all_methods.tex").write_text("\n".join(L) + "\n",
                                                  encoding="utf-8")
    print(f"[OK] {args.out / 'uc1_all_methods.tex'}")

    for ds in args.datasets:
        print(f"\n--- {ds} (paired n={pair_n.get(ds, 0)}) ---")
        for m, lab, _e in ROWS:
            if m == "group":
                print(f"  [{lab}]")
                continue
            r = idx.get((ds, m))
            if not r or "plain_f1" not in r:
                print(f"    {lab:42s} --")
                continue
            a = r.get("anon_f1")
            print(f"    {lab:42s} F1={r['plain_f1']:.3f}"
                  + (f"  anon={a:.3f}  delta={r['delta_f1_anon']:+.3f}"
                     if a is not None else ""))


if __name__ == "__main__":
    main()
