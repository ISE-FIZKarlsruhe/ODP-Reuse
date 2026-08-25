"""RQ2: how much of each reuse form's occurrences the largest contributor supplies.

Table~\\ref{tab:rq2_forms} gives micro, macro, median and $n_{>0}$ per form.
Those four separate a form used by nearly every ontology from one confined to a
minority, but they do not separate a form spread evenly over that minority from
one that a single member of it dominates: $n_{>0}$ counts contributors and says
nothing about how unequally they contribute. This adds the missing statistic --
the share of a form's occurrences supplied by its largest contributing ontology,
and by its largest three -- so the paper can give a worked example rather than
only naming the gap.

Reads the same pairs.csv the table is built from, so every micro / macro /
median / n>0 figure it prints is recomputed here and can be checked against
Table~\\ref{tab:rq2_forms} directly.

    python pipeline/rq2_concentration.py
"""
from __future__ import annotations

import csv
import pathlib
import statistics
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAIRS = ROOT / "pipeline/out/rq_iri_gated/pairs.csv"

ROLES = ["direct", "subclass_of_pattern", "superclass_of_pattern",
         "subproperty_of_pattern", "superproperty_of_pattern",
         "used_as_property", "in_restriction", "equivalence"]

# The names the paper uses, so the output can be read straight into the text.
PAPER = {"direct": "direct IRI use",
         "subclass_of_pattern": "subclass of pattern",
         "superclass_of_pattern": "superclass of pattern",
         "subproperty_of_pattern": "subproperty of pattern",
         "superproperty_of_pattern": "superproperty of pattern",
         "used_as_property": "property as predicate",
         "in_restriction": "in restriction",
         "equivalence": "equivalence"}

rows = list(csv.DictReader(PAIRS.open(encoding="utf-8")))

for ds, label in (("github", "GitHub"), ("noOBO", "Paper & Repository")):
    sub = [r for r in rows if r["dataset"] == ds]

    per = defaultdict(lambda: {k: 0 for k in ROLES})     # ontology -> counts
    tot = {k: 0 for k in ROLES}
    top_pair = {k: (0, "", "") for k in ROLES}
    for r in sub:
        for k in ROLES:
            c = int(r["role_" + k] or 0)
            tot[k] += c
            per[r["ontology"]][k] += c
            if c > top_pair[k][0]:
                top_pair[k] = (c, r["ontology"], r["pattern"])
    T = sum(tot.values())

    shares = defaultdict(list)
    for o, c in per.items():
        s = sum(c.values())
        if s == 0:
            continue
        for k in ROLES:
            shares[k].append(100 * c[k] / s)
    n_onto = len(shares[ROLES[0]])

    print("=" * 96)
    print(f"{label}  ({n_onto} ontologies, {len(sub)} pairs, "
          f"{T:,} role occurrences)")
    print("=" * 96)
    print(f"  {'form':<26}{'micro':>7}{'macro':>7}{'med':>6}{'n>0':>6}"
          f"{'occ':>8}{'top1':>7}{'top3':>7}{'top pair':>10}"
          f"{'micro-1pair':>13}")
    for k in ROLES:
        by_o = Counter({o: c[k] for o, c in per.items() if c[k]})
        occ = tot[k]
        if not occ:
            print(f"  {PAPER[k]:<26}{'--':>7}{'--':>7}{'--':>6}"
                  f"{len(by_o):>6}{0:>8}{'--':>7}{'--':>7}{'--':>10}{'--':>13}")
            continue
        ranked = by_o.most_common()
        top1 = 100 * ranked[0][1] / occ
        top3 = 100 * sum(v for _, v in ranked[:3]) / occ
        tp = 100 * top_pair[k][0] / occ
        # What the headline micro share becomes if the single largest pair goes.
        micro_less = 100 * (occ - top_pair[k][0]) / (T - top_pair[k][0])
        print(f"  {PAPER[k]:<26}"
              f"{100*occ/T:6.2f}%{statistics.mean(shares[k]):6.1f}%"
              f"{statistics.median(shares[k]):5.1f}%{len(by_o):>6}"
              f"{occ:>8}{top1:6.1f}%{top3:6.1f}%{tp:9.1f}%"
              f"{micro_less:12.2f}%")

    print()
    print("  largest contributor per form:")
    for k in ROLES:
        by_o = Counter({o: c[k] for o, c in per.items() if c[k]})
        if not by_o:
            print(f"    {PAPER[k]:<26} none")
            continue
        o, v = by_o.most_common(1)[0]
        print(f"    {PAPER[k]:<26} {v:>6} of {tot[k]:>6} "
              f"({100*v/tot[k]:5.1f}%)  {o}")
    print()

print("Columns: micro/macro/med/n>0 reproduce Table 4. occ = the form's total")
print("role occurrences. top1/top3 = the share of those occurrences supplied by")
print("the largest one / three contributing ontologies. top pair = the share")
print("supplied by the single largest ontology-pattern pair. micro-1pair = what")
print("the form's micro share becomes with that one pair removed.")
