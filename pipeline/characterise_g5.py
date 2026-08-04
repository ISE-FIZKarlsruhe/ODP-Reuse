#!/usr/bin/env python3
"""characterise_g5.py -- why do 424 substantive ontologies share no entity
with any of their patterns?

Reviewer 5 of the earlier version observed that the analysis yields "a lot of
'what'" while the question most needing an answer is the "why not". The 424
ontologies removed by rule G5 -- they parse, are unique, contribute entities
of their own, and were found BY a pattern-IRI search, yet overlap no pattern
-- are the natural evidence for that question, and the earlier version only
counted them.

This partitions them into named causes, all decidable from artefacts already
on disk, so the answer is a measurement rather than a conjecture:

  remodelled_confirmed  a human adjudicator confirmed the ontology reuses the
                        pattern under its own namespace (from the 331-item
                        label adjudication). These are genuine reuse that
                        exact-IRI matching cannot see.
  remodelled_candidate  normalised-label matching proposes reuse but the
                        adjudicator rejected it, or the pair was in the
                        deferred stratum. Mostly name collisions.
  namespace_only        the ontology declares or imports a pattern namespace
                        but uses none of its entities in any axiom -- the
                        "declared but unused" case.
  no_pattern_entities   every candidate pattern for this ontology has an empty
                        entity set after normalisation, so no overlap was
                        decidable either way.
  disjoint              the ontology genuinely uses none of the pattern's
                        entities and makes no reference to its namespace: a
                        false positive of the discovery search.

Usage:
    python pipeline/characterise_g5.py --manifest pipeline/out/MANIFEST.csv \
        --adjudication pipeline/out/adjudication/human_sample.csv \
        --out pipeline/out/g5
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--adjudication", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    g5 = [r for r in rows if (r["drop_reason"] or "").startswith("G5_has_gt")]

    # label-adjudication outcomes, keyed by (dataset, ontology)
    confirmed, proposed = set(), set()
    with a.adjudication.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            k = (r["dataset"], r["ontology"])
            proposed.add(k)
            if r.get("human_verdict", "").strip().lower() == "reuse":
                confirmed.add(k)

    cause, per_ds = {}, defaultdict(Counter)
    for r in g5:
        k = (r["dataset"], r["ontology_id"])
        n_pat = int(r["n_pattern_files"] or 0)
        n_err = int(r["n_pattern_errors"] or 0)
        declared = (r["declared_iris"] or "")
        # a pattern-library namespace mentioned in the ontology's own header
        ns_hit = any(s in declared.lower() for s in
                     ("ontologydesignpatterns.org", "/cp/owl/", "dead_simple",
                      "gong.manchester", "industrial-standard"))
        if k in confirmed:
            c = "remodelled_confirmed"
        elif k in proposed:
            c = "remodelled_candidate"
        elif n_pat and n_err >= n_pat:
            c = "no_pattern_entities"
        elif ns_hit:
            c = "namespace_only"
        else:
            c = "disjoint"
        cause[k] = c
        per_ds[r["dataset"]][c] += 1

    order = ["remodelled_confirmed", "remodelled_candidate", "namespace_only",
             "no_pattern_entities", "disjoint"]
    a.out.mkdir(parents=True, exist_ok=True)
    p = a.out / "g5_causes.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "cause", "n", "pct_of_dataset_g5"])
        for ds in sorted(per_ds):
            tot = sum(per_ds[ds].values())
            for c in order:
                if per_ds[ds][c]:
                    w.writerow([ds, c, per_ds[ds][c],
                                round(100 * per_ds[ds][c] / tot, 1)])
    with (a.out / "g5_per_ontology.csv").open("w", newline="",
                                              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "ontology_id", "cause"])
        for (ds, o), c in sorted(cause.items()):
            w.writerow([ds, o, c])

    total = Counter(cause.values())
    n = len(g5)
    print(f"G5 removals characterised : {n}")
    print()
    for c in order:
        if total[c]:
            print(f"  {c:<22} {total[c]:>4}  ({100*total[c]/n:4.1f}%)")
    print()
    for ds in sorted(per_ds):
        tot = sum(per_ds[ds].values())
        parts = ", ".join(f"{c}={per_ds[ds][c]}" for c in order if per_ds[ds][c])
        print(f"  {ds:<8} n={tot:<4} {parts}")
    print(f"\n[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
