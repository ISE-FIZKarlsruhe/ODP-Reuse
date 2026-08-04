"""How much of the CF feature is the answer?

UC2's ontology-kNN collaborative filter measures similarity as Jaccard over
`Ontology.tokens` -- the normalised bag of local names and labels of the
ontology's entities. There is no label leakage: `recommend()` never sees the
test ontology's `gt_stems`.

But an ontology that reuses a pattern BY IRI necessarily contains that
pattern's entity local names among its tokens. So the feature the neighbour
search runs on partly *encodes* the ground truth. This quantifies how much,
which is the number a reviewer will ask for.

Method: for each gate-passing ontology, take its WIDE entity set from the
released pairs/manifest data and its reused patterns' entity sets, normalise
local names the same way recobench does, and report what share of the
ontology's token bag is contributed by the patterns it reuses.
"""
import collections
import csv
import pathlib
import re
import statistics

ROOT = pathlib.Path(r"C:\Users\eno\Documents\PhD\odp-reuse-benchmark")
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_\-]+")


def localname(iri: str) -> str:
    s = iri.rstrip("/#")
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def toks(iri: str) -> set[str]:
    ln = localname(iri)
    return {t.lower() for t in CAMEL.split(ln) if t and not t.isdigit()}


pairs = [r for r in csv.DictReader(
    (ROOT / "pipeline/out/rq_iri_gated/pairs.csv").open(encoding="utf-8"))
    if int(r["n_reused"] or 0) > 0 and r["dataset"] == "github"]

# per ontology: how many entities it has, how many are pattern entities it reused
per = collections.defaultdict(lambda: {"n_pat_ent": 0, "reused": 0, "pats": set()})
for r in pairs:
    d = per[r["ontology"]]
    d["n_pat_ent"] += int(r["n_pattern_entities"] or 0)
    d["reused"] += int(r["n_reused"] or 0)
    d["pats"].add(r["pattern"])

man = {r["ontology_id"]: r for r in csv.DictReader(
    (ROOT / "pipeline/out/MANIFEST.csv").open(encoding="utf-8"))
    if r["dataset"] == "github"}

shares, n_pats = [], []
for o, d in per.items():
    m = man.get(o)
    if not m:
        continue
    wide = int(m["n_entities_wide"] or 0)
    if wide == 0:
        continue
    # entities of this ontology that ARE reused pattern entities
    shares.append(min(1.0, d["reused"] / wide))
    n_pats.append(len(d["pats"]))

print(f"gated GitHub ontologies analysed : {len(shares)}")
print()
print("Share of an ontology's WIDE entity set that consists of entities it")
print("reused from a pattern -- i.e. the part of the CF feature that is")
print("literally the pattern's own vocabulary:")
print(f"  mean   : {statistics.mean(shares)*100:.1f}%")
print(f"  median : {statistics.median(shares)*100:.1f}%")
q = statistics.quantiles(shares, n=4)
print(f"  quartiles : {q[0]*100:.1f}% / {q[1]*100:.1f}% / {q[2]*100:.1f}%")
print(f"  >=25% of the file is reused pattern vocabulary : "
      f"{sum(1 for s in shares if s >= .25)} ontologies "
      f"({sum(1 for s in shares if s >= .25)/len(shares)*100:.0f}%)")
print(f"  >=50% : {sum(1 for s in shares if s >= .50)} "
      f"({sum(1 for s in shares if s >= .50)/len(shares)*100:.0f}%)")
print()
print(f"mean reused patterns per ontology : {statistics.mean(n_pats):.2f}")
print(f"ontologies reusing exactly 1 pattern : "
      f"{sum(1 for n in n_pats if n == 1)} "
      f"({sum(1 for n in n_pats if n==1)/len(n_pats)*100:.0f}%)")
