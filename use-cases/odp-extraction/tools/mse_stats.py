#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mse_stats.py -- what can be said about extracted patterns when there is no
answer key?

On ODP-Reuse a predicted cluster is scored against a verified pattern instance.
On a corpus nobody has annotated -- the materials-science collection here -- no
such score exists, and inventing one would be worse than reporting none. What
*can* be measured are properties that make a proposed pattern more or less
worth a domain expert's time. Six are computed, and each is a claim that can be
checked without knowing the answer:

A. YIELD          how many groups each method proposes, how large they are, and
                  what share of an ontology's entities they touch. A method
                  proposing 300 groups per ontology is not usable by a human
                  reviewer whatever its quality.

B. CROSS-METHOD   independent methods agreeing on a group is evidence the group
   AGREEMENT      is a real structure rather than an artefact of one algorithm.
                  Every pair of methods is matched with the same Hungarian
                  entity-F1 used for scoring against ground truth -- here method
                  against method. A group corroborated by several methods is the
                  first thing to put in front of an expert.

C. RECURRENCE     a design pattern is by definition a structure that recurs. A
                  proposed group that appears, up to renaming, in several
                  ontologies has evidence no single-ontology score can give.
                  Groups are compared across ontologies by shared entity IRIs
                  and, separately, by shared local names, so both literal reuse
                  and independent re-modelling are visible.

D. FOUNDATIONAL   a group whose entities specialise a foundational or
   ANCHORING      mid-level ontology (BFO, EMMO, IAO, RO, PMDco, CCO) is more
                  likely a modelling pattern than an incidental clique. The
                  share of anchored groups is reported per method.

E. KNOWN-ODP      a group overlapping a pattern from the ODP-Reuse reference
   CONTACT        libraries is a *recovered* known pattern -- a validation
                  signal. A group overlapping none is a candidate novel one.
                  Both counts matter, and they matter differently.

F. STRUCTURE      per group: size, class/property mix, and whether the entities
                  are connected in the ontology. A disconnected group is
                  unlikely to be a pattern whatever else it scores.

Together these rank the proposed groups for expert review, which is what the
evaluation platform consumes. None of them is a substitute for the expert
judgement; they decide what the expert is asked about.

Usage
-----
    python tools/mse_stats.py --predictions results/mseo/predictions.jsonl \\
        --corpus corpora/mseo --out stats/mseo \\
        --patterns-root <MultiSource>/patterns-repository
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
_REPO = _USECASE.parent.parent
sys.path.insert(0, str(_USECASE))
sys.path.insert(0, str(_REPO / "pipeline"))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

# Foundational and mid-level vocabularies an MSE pattern may specialise. Any
# entity IRI containing one of these markers counts as an anchor.
FOUNDATIONAL_MARKERS = {
    "BFO": ("obolibrary.org/obo/BFO_", "purl.obolibrary.org/obo/bfo"),
    "EMMO": ("emmo.info/emmo", "w3id.org/emmo"),
    "IAO": ("obolibrary.org/obo/IAO_",),
    "RO": ("obolibrary.org/obo/RO_",),
    "PMDco": ("w3id.org/pmd/co", "materialdigital.de",),
    "CCO": ("CommonCoreOntologies", "www.ontologyrepository.com",),
    "QUDT": ("qudt.org",),
    "SKOS": ("w3.org/2004/02/skos",),
    "DUL/DOLCE": ("ontologydesignpatterns.org/ont/dul", "DOLCE"),
}


def local_name(iri: str) -> str:
    cut = max(iri.rfind("#"), iri.rfind("/"))
    return iri[cut + 1:] if cut >= 0 else iri


def f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    p, r = inter / len(a), inter / len(b)
    return 2 * p * r / (p + r)


def best_match_f1(ca: list[set], cb: list[set]) -> float:
    """Hungarian mean F1 between two cluster sets -- the ground-truth metric,
    with one method's output standing in for the answer key."""
    if not ca or not cb:
        return 0.0
    m = np.zeros((len(ca), len(cb)))
    for i, a in enumerate(ca):
        for j, b in enumerate(cb):
            m[i, j] = f1(a, b)
    rows, cols = linear_sum_assignment(1.0 - m)
    return float(np.mean([m[i, j] for i, j in zip(rows, cols)]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--patterns-root", type=Path, default=None,
                    help="a directory of reference ODP files; groups touching "
                         "one are recorded as recovering a known pattern")
    ap.add_argument("--min-size", type=int, default=2,
                    help="groups smaller than this are excluded from the "
                         "recurrence and review analyses: a one-entity group "
                         "carries no structure to recur")
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A METHOD'S POPULATION MUST COME FROM ONE RUN.
    #
    # Several runs can emit the same method name: the full conceptual-components
    # pipeline re-emits its own step 1 beside the merged reading, and that is the
    # method the structural run already produced. Different runs apply different
    # exclusion rules, so merging them mixes populations. Deduplicating per
    # (method, ontology) is not enough -- it still lets an ontology one run
    # deliberately excluded enter through another. That happened here: the
    # structural run caps conceptual-component extraction at 8,000 entities and
    # so skips PATO, the full pipeline applies no cap, and PATO alone contributed
    # 4,298 of 9,502 groups, lifting the mean from 51.0 to 92.3 on the strength
    # of one artefact.
    #
    # So the rule is per METHOD: the first file to supply a method owns that
    # method entirely, and later files' rows for it are dropped whole. Every
    # drop is reported.
    rows: list[dict] = []
    owner: dict[str, str] = {}
    dropped: Counter = Counter()
    for p in args.predictions:
        src = p.parent.name
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            m = r["method"]
            owner.setdefault(m, src)
            if owner[m] != src:
                dropped[f'{m} from {src} (owned by {owner[m]})'] += 1
                continue
            r["_source"] = src
            rows.append(r)
    print(f"{len(rows)} (ontology, method) prediction rows")
    print("  method -> source run: "
          + ", ".join(f"{m}={s}" for m, s in sorted(owner.items())))
    if dropped:
        print("  rows dropped so that each method has ONE population:")
        for k, v in dropped.most_common():
            print(f"    {v:5d}  {k}")

    by_method: dict[str, dict[str, list[set]]] = defaultdict(dict)
    for r in rows:
        by_method[r["method"]][r["ontology"]] = [set(c) for c in r["clusters"]]
    methods = sorted(by_method)
    ontologies = sorted({r["ontology"] for r in rows})
    print(f"{len(methods)} methods, {len(ontologies)} ontologies")

    out: dict = {"n_methods": len(methods), "n_ontologies": len(ontologies),
                 "methods": methods}

    # ---------------- A: yield ----------------------------------------
    yields = []
    for m in methods:
        per = by_method[m]
        counts = [len(v) for v in per.values()]
        sizes = [len(c) for v in per.values() for c in v]
        cov = []
        for r in rows:
            if r["method"] != m or not r.get("n_entities"):
                continue
            touched = len(set().union(*[set(c) for c in r["clusters"]])
                          ) if r["clusters"] else 0
            cov.append(touched / r["n_entities"])
        yields.append({
            "method": m, "n_ontologies": len(per),
            "groups_per_ontology_mean": round(st.mean(counts), 2) if counts else 0,
            "groups_per_ontology_median": round(st.median(counts), 1) if counts else 0,
            "group_size_mean": round(st.mean(sizes), 2) if sizes else 0,
            "group_size_median": round(st.median(sizes), 1) if sizes else 0,
            "groups_ge_min_size": sum(1 for s in sizes if s >= args.min_size),
            "entity_coverage_mean": round(st.mean(cov), 4) if cov else 0,
        })
    out["A_yield"] = yields

    # ---------------- B: cross-method agreement -----------------------
    agree = []
    for i, ma in enumerate(methods):
        for mb in methods[i + 1:]:
            common = set(by_method[ma]) & set(by_method[mb])
            if not common:
                continue
            vals = [best_match_f1(by_method[ma][o], by_method[mb][o])
                    for o in sorted(common)]
            agree.append({"method_a": ma, "method_b": mb,
                          "n_ontologies": len(common),
                          "mean_best_match_f1": round(float(np.mean(vals)), 4)})
    agree.sort(key=lambda d: -d["mean_best_match_f1"])
    out["B_cross_method_agreement"] = agree

    # ---------------- C: recurrence across ontologies ------------------
    # Two views: shared IRIs (literal reuse) and shared local names
    # (independent re-modelling of the same idea).
    def recurrence(key_fn, label):
        groups = []
        for m in methods:
            for onto, cl in by_method[m].items():
                for gi, g in enumerate(cl):
                    if len(g) < args.min_size:
                        continue
                    groups.append((m, onto, gi, frozenset(key_fn(x) for x in g)))
        index: dict[frozenset, list] = defaultdict(list)
        # exact-key recurrence: identical key sets across different ontologies
        for m, onto, gi, keys in groups:
            index[keys].append((m, onto, gi))
        recur = [(k, v) for k, v in index.items()
                 if len({o for _m, o, _g in v}) > 1]
        recur.sort(key=lambda kv: -len({o for _m, o, _g in kv[1]}))
        return {
            "view": label,
            "n_groups_considered": len(groups),
            "n_recurrent_keysets": len(recur),
            "pct_groups_recurrent": round(
                100 * sum(len(v) for _k, v in recur) / max(len(groups), 1), 2),
            "top": [{"n_ontologies": len({o for _m, o, _g in v}),
                     "n_occurrences": len(v),
                     "size": len(k),
                     "members": sorted(k)[:12],
                     "methods": sorted({m for m, _o, _g in v})}
                    for k, v in recur[:args.top]],
        }

    out["C_recurrence_by_iri"] = recurrence(lambda x: x, "shared entity IRIs")
    out["C_recurrence_by_localname"] = recurrence(
        lambda x: local_name(x).lower(), "shared local names")

    # ---------------- D: foundational anchoring ------------------------
    anchor = []
    for m in methods:
        tot = anch = 0
        per_vocab = Counter()
        for cl in by_method[m].values():
            for g in cl:
                if len(g) < args.min_size:
                    continue
                tot += 1
                hit = {v for v, marks in FOUNDATIONAL_MARKERS.items()
                       if any(mk.lower() in e.lower() for e in g for mk in marks)}
                if hit:
                    anch += 1
                    for v in hit:
                        per_vocab[v] += 1
        anchor.append({"method": m, "groups": tot, "anchored": anch,
                       "pct_anchored": round(100 * anch / max(tot, 1), 2),
                       "by_vocabulary": dict(per_vocab.most_common())})
    out["D_foundational_anchoring"] = anchor

    # ---------------- E: contact with known ODPs -----------------------
    if args.patterns_root and args.patterns_root.is_dir():
        from odpqual.entities import collect_entities, parse_graph, strip_annotations
        known: dict[str, set[str]] = {}
        for pf in sorted(args.patterns_root.rglob("*")):
            if pf.suffix.lower() not in (".owl", ".ttl", ".rdf", ".xml"):
                continue
            pg, _ = parse_graph(pf)
            if pg is None:
                continue
            pg, _ = strip_annotations(pg)
            ents = set(collect_entities(pg, wide=True))
            if ents:
                known[pf.stem] = ents
        allknown = set().union(*known.values()) if known else set()
        contact = []
        for m in methods:
            tot = hit = 0
            names = Counter()
            for cl in by_method[m].values():
                for g in cl:
                    if len(g) < args.min_size:
                        continue
                    tot += 1
                    if g & allknown:
                        hit += 1
                        for pname, pents in known.items():
                            if g & pents:
                                names[pname] += 1
            contact.append({"method": m, "groups": tot,
                            "touching_known_odp": hit,
                            "pct": round(100 * hit / max(tot, 1), 2),
                            "top_patterns": dict(names.most_common(10))})
        out["E_known_odp_contact"] = {"n_reference_patterns": len(known),
                                      "per_method": contact}
    else:
        out["E_known_odp_contact"] = {"note": "no --patterns-root given"}

    # ---------------- F: review shortlist ------------------------------
    # What an expert should be asked about first: groups of a reviewable size,
    # corroborated by more than one method, recurring across ontologies.
    shortlist = []
    name_index: dict[frozenset, list] = defaultdict(list)
    for m in methods:
        for onto, cl in by_method[m].items():
            for gi, g in enumerate(cl):
                if not (args.min_size <= len(g) <= 25):
                    continue
                name_index[frozenset(local_name(x).lower() for x in g)].append(
                    (m, onto, gi, g))
    for keys, occ in name_index.items():
        onts = {o for _m, o, _g, _s in occ}
        meths = {m for m, _o, _g, _s in occ}
        if len(onts) < 2 and len(meths) < 2:
            continue
        example = occ[0][3]
        shortlist.append({
            "size": len(keys),
            "n_ontologies": len(onts),
            "n_methods": len(meths),
            "methods": sorted(meths),
            "ontologies": sorted(onts)[:10],
            "local_names": sorted(keys)[:15],
            "example_iris": sorted(example)[:15],
            "priority": len(onts) * 2 + len(meths),
        })
    shortlist.sort(key=lambda d: (-d["priority"], -d["n_ontologies"]))
    out["F_review_shortlist"] = {
        "n_candidates": len(shortlist),
        "top": shortlist[:args.top],
    }

    (args.out / "mse_stats.json").write_text(json.dumps(out, indent=2),
                                             encoding="utf-8")
    with (args.out / "yield.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(yields[0]))
        w.writeheader()
        w.writerows(yields)
    with (args.out / "agreement.csv").open("w", newline="", encoding="utf-8") as fh:
        if agree:
            w = csv.DictWriter(fh, fieldnames=list(agree[0]))
            w.writeheader()
            w.writerows(agree)

    print(f"\n[OK] {args.out / 'mse_stats.json'}")
    print("\nA. yield")
    for y in yields:
        print(f"  {y['method']:22s} groups/onto={y['groups_per_ontology_mean']:7.2f} "
              f"size_med={y['group_size_median']:5.1f} "
              f"coverage={y['entity_coverage_mean']:.3f}")
    print("\nB. cross-method agreement (top 6)")
    for a in agree[:6]:
        print(f"  {a['method_a']:20s} vs {a['method_b']:20s} "
              f"F1={a['mean_best_match_f1']:.3f}")
    print(f"\nC. recurrence: {out['C_recurrence_by_localname']['n_recurrent_keysets']} "
          f"local-name groups recur across ontologies "
          f"({out['C_recurrence_by_localname']['pct_groups_recurrent']}% of groups)")
    print(f"F. review shortlist: {len(shortlist)} candidates")


if __name__ == "__main__":
    main()
