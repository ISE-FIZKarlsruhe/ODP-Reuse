"""The reuse evidence, keyed the way the pattern libraries can be joined to it.

``pipeline/out/rq_iri_gated/pairs.csv`` identifies a pattern by *stem* only --
there is no library column and no pattern IRI -- so any join between the reuse
evidence and the libraries is a join on case-folded stem.  That is a real
limitation, not a convenience: a stem present in two libraries cannot be
attributed to one of them from ``pairs.csv`` alone.  ``ambiguous_stems()``
reports exactly which stems are affected so the size of the problem is stated
rather than assumed.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


def load_pairs(pairs_csv: Path) -> list[dict]:
    with pairs_csv.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def reuse_by_stem(pairs: list[dict]) -> dict[str, dict]:
    """case-folded stem -> aggregate reuse evidence."""
    agg: dict[str, dict] = {}
    for r in pairs:
        k = r["pattern"].casefold()
        d = agg.setdefault(k, {
            "pattern": r["pattern"],
            "n_pairs": 0,
            "n_ontologies": 0,
            "n_complete": 0,
            "by_dataset": Counter(),
            "ontologies": set(),
            "coverage": [],
        })
        d["n_pairs"] += 1
        d["by_dataset"][r["dataset"]] += 1
        d["ontologies"].add((r["dataset"], r["ontology"]))
        if r.get("complete") == "1":
            d["n_complete"] += 1
        try:
            d["coverage"].append(float(r.get("coverage_pct") or 0.0))
        except ValueError:
            pass
    for d in agg.values():
        d["n_ontologies"] = len(d["ontologies"])
    return agg


def ambiguous_stems(records: list[dict]) -> dict[str, list[str]]:
    """case-folded stem -> libraries containing it, for stems in >1 library."""
    m: dict[str, set[str]] = defaultdict(set)
    for r in records:
        m[r["stem"].casefold()].add(r["library"])
    return {k: sorted(v) for k, v in m.items() if len(v) > 1}
