"""RQ2: are the reuse-form shares micro or macro averages, and does it matter?

The paper reports shares over pooled role occurrences (micro). A reviewer
recomputed them per ontology and averaged (macro) and found direct
instantiation rises to ~45%, back inside the 38-63% band the paper claims to
refute. This checks that independently from pairs.csv.
"""
import csv
import pathlib
import statistics
from collections import Counter, defaultdict

ROOT = pathlib.Path(r"C:\Users\eno\Documents\PhD\odp-reuse-benchmark")
ROLES = ["direct", "subclass_of_pattern", "superclass_of_pattern",
         "subproperty_of_pattern", "superproperty_of_pattern",
         "used_as_property", "in_restriction", "equivalence"]

rows = list(csv.DictReader(
    (ROOT / "pipeline/out/rq_iri_gated/pairs.csv").open(encoding="utf-8")))

for ds in ("github", "noOBO"):
    sub = [r for r in rows if r["dataset"] == ds]
    with_overlap = [r for r in sub if int(r["n_reused"] or 0) > 0]
    print(f"===== {ds}")
    print(f"  all harvested pairs      : {len(sub)}")
    print(f"  pairs with >=1 overlap   : {len(with_overlap)}")

    # ---- MICRO over ALL pairs (what the released CSV does)
    tot = {k: 0 for k in ROLES}
    for r in sub:
        for k in ROLES:
            tot[k] += int(r["role_" + k] or 0)
    T = sum(tot.values())
    print(f"  total role occurrences   : {T:,}")
    print()
    print(f"  {'role':<26}{'MICRO':>8}{'MACRO mean':>12}{'MACRO med':>11}"
          f"{'onto with >0':>14}")

    # ---- MACRO per ontology
    per = defaultdict(lambda: {k: 0 for k in ROLES})
    for r in sub:
        for k in ROLES:
            per[r["ontology"]][k] += int(r["role_" + k] or 0)
    shares = defaultdict(list)
    nonzero = Counter()
    n_onto = 0
    for o, c in per.items():
        s = sum(c.values())
        if s == 0:
            continue
        n_onto += 1
        for k in ROLES:
            shares[k].append(100 * c[k] / s)
            if c[k] > 0:
                nonzero[k] += 1

    for k in ROLES:
        micro = 100 * tot[k] / T if T else 0
        print(f"  {k:<26}{micro:7.1f}%{statistics.mean(shares[k]):11.1f}%"
              f"{statistics.median(shares[k]):10.1f}%"
              f"{nonzero[k]:>9}/{n_onto}")
    print()

    # ---- concentration of used_as_property (the contested claim)
    if ds == "github":
        uap = [(int(r["role_used_as_property"] or 0), r["ontology"], r["pattern"])
               for r in sub]
        uap = [x for x in uap if x[0] > 0]
        tot_uap = sum(x[0] for x in uap)
        uap.sort(reverse=True)
        print(f"  used_as_property concentration ({tot_uap:,} occurrences):")
        print(f"    pairs with any      : {len(uap)}/{len(sub)} "
              f"({len(uap)/len(sub)*100:.1f}%)")
        by_o = Counter()
        for n, o, p in uap:
            by_o[o] += n
        print(f"    ontologies with any : {len(by_o)}/{n_onto} "
              f"({len(by_o)/n_onto*100:.1f}%)")
        print(f"    top pair            : {uap[0][0]:,} = "
              f"{uap[0][0]/tot_uap*100:.1f}% of all")
        top_o = by_o.most_common(3)
        print(f"    top ontology        : {top_o[0][1]:,} = "
              f"{top_o[0][1]/tot_uap*100:.1f}% of all")
        print(f"    top 3 ontologies    : "
              f"{sum(v for _, v in top_o)/tot_uap*100:.1f}% of all")
        # leave-one-pair-out on the headline
        micro_uap = 100 * tot["used_as_property"] / T
        T2 = T - uap[0][0]
        m2 = 100 * (tot["used_as_property"] - uap[0][0]) / T2
        print(f"    headline {micro_uap:.1f}% -> {m2:.1f}% "
              f"if the single largest pair is dropped")
        print()
