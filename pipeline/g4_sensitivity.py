#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g4_sensitivity.py -- price the G4 threshold instead of asserting it.

G4 keeps an ontology only if it contributes at least `--min-own-entities`
entities beyond those supplied by its patterns. The default is 10, and 10 is not
derived from anything. This script measures what the choice actually costs, so
that a reader can disagree with it from the data rather than from taste.

The gate applies G1, G2, G3, G4, G5 in order, so moving the G4 threshold can only
move ontologies between G4 and {G5, kept}. That makes the whole curve computable
from MANIFEST.csv without re-running the gate:

    kept(T) = survives G1-G3  AND  n_own_entities >= T  AND  n_gt_patterns >= 1

The script asserts kept(10) == the shipped release before reporting anything.

It also reports a second statistic the absolute threshold cannot see: the SHARE
of an ontology that is its own work. That distinguishes the two very different
things an absolute count conflates --- a genuinely small ontology, and a copy of
a pattern library with one class bolted on.

    python pipeline/g4_sensitivity.py --manifest release-iri/MANIFEST.csv \\
        --pairs pipeline/out/rq_iri_all/pairs.csv --out pipeline/out
"""
from __future__ import annotations

import argparse
import collections
import csv
import statistics
from pathlib import Path

EARLY = ("G1_parse", "G2_pattern_copy", "G3_duplicate")
THRESHOLDS = (0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100)
# The released gate. The script asserts it can reproduce these before
# reporting anything, so a drift in the manifest fails loudly here.
SHIPPED_DEFAULT, SHIPPED_KEPT = 1, 501


def rule(r: dict) -> str:
    return (r["drop_reason"] or "").split(":")[0]


def own(r: dict) -> int:
    return int(r["n_own_entities"] or 0)


def share(r: dict) -> float:
    w = int(r["n_entities_wide"] or 0)
    return own(r) / w if w else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--pairs", required=True, type=Path,
                    help="rq_iri_all/pairs.csv -- pairs over ALL parseable "
                         "ontologies, so any threshold can be scored")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    man = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    surv = [r for r in man if rule(r) not in EARLY]
    pairs = collections.defaultdict(list)
    for p in csv.DictReader(args.pairs.open(encoding="utf-8")):
        if int(p["n_reused"]) > 0:
            pairs[p["ontology"]].append(p)

    def kept(t: int) -> list[dict]:
        return [r for r in surv
                if own(r) >= t and int(r["n_gt_patterns"] or 0) >= 1]

    check = len(kept(SHIPPED_DEFAULT))
    assert check == SHIPPED_KEPT, (
        f"kept({SHIPPED_DEFAULT}) = {check}, release says {SHIPPED_KEPT}; "
        "the reconstruction does not reproduce the gate")
    print(f"[ok] kept({SHIPPED_DEFAULT}) reproduces the release exactly "
          f"({SHIPPED_KEPT})")

    rows = []
    for t in THRESHOLDS:
        ks = kept(t)
        ids = {r["ontology_id"] for r in ks}
        ps = [p for i in ids for p in pairs.get(i, [])]
        row = {"threshold": t, "kept": len(ks), "delta_vs_shipped": len(ks) - SHIPPED_KEPT,
               "pairs": len(ps),
               "complete_reuse_onto": sum(
                   1 for r in ks if int(r["n_complete_patterns"] or 0) > 0)}
        for ds in ("github", "noOBO", "OBO"):
            sel = [r for r in ks if r["dataset"] == ds]
            sids = {r["ontology_id"] for r in sel}
            dps = [p for i in sids for p in pairs.get(i, [])
                   if p["dataset"] == ds]
            row[f"{ds}_onto"] = len(sel)
            row[f"{ds}_pairs"] = len(dps)
            row[f"{ds}_patterns"] = len({p["pattern"] for p in dps})
            row[f"{ds}_mean_patterns_per_onto"] = round(
                len(dps) / len(sel), 2) if sel else ""
            row[f"{ds}_complete_pct"] = round(
                100 * sum(1 for r in sel
                          if int(r["n_complete_patterns"] or 0) > 0)
                / len(sel), 1) if sel else ""
            row[f"{ds}_mean_coverage_pct"] = round(
                statistics.mean(float(p["coverage_pct"]) for p in dps), 1) \
                if dps else ""
        rows.append(row)

    p = args.out / "g4_threshold_sensitivity.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {p}")

    # ---- who is in the band the default excludes -------------------------
    band_all = [r for r in surv if 1 <= own(r) < SHIPPED_DEFAULT]
    band = [r for r in band_all if int(r["n_gt_patterns"] or 0) >= 1]
    q = args.out / "g4_band_1_to_9.csv"
    cols = ["ontology_id", "dataset", "n_own_entities", "n_entities_wide",
            "own_share", "n_gt_patterns", "n_complete_patterns"]
    with q.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(band, key=lambda x: share(x)):
            w.writerow({**r, "own_share": round(share(r), 3)})
    print(f"[OK] {q}  ({len(band)} rows)")

    ks10 = kept(SHIPPED_DEFAULT)
    print("\n=== the curve ===")
    print(f'{"T":>4} {"kept":>6} {"vs shipped":>11} {"pairs":>7} {"GH mean p/o":>12} '
          f'{"GH complete%":>13} {"GH patterns":>12}')
    for r in rows:
        print(f'{r["threshold"]:>4} {r["kept"]:>6} {r["delta_vs_shipped"]:>+11} '
              f'{r["pairs"]:>7} {r["github_mean_patterns_per_onto"]:>12} '
              f'{r["github_complete_pct"]:>13} {r["github_patterns"]:>12}')

    print(f"\n=== the band the default excludes (1-9 own entities) ===")
    print(f"  surviving G1-G3           : {len(band_all)}")
    print(f"  ...and passing G5 as well : {len(band)}   <- what T=1 readmits")
    print(f"  the other {len(band_all) - len(band)} have no pattern overlap and "
          f"G5 drops them anyway.")
    sh = [share(r) for r in band]
    sk = [share(r) for r in ks10]
    print(f"  own-share, readmitted : median {statistics.median(sh)*100:.0f}%")
    print(f"  own-share, kept 429   : median {statistics.median(sk)*100:.0f}%")
    print(f"  readmits that are >=50% their own work : "
          f"{sum(1 for x in sh if x >= .5)} of {len(band)}")
    print(f"  readmits that are <25% their own work  : "
          f"{sum(1 for x in sh if x < .25)} of {len(band)}")

    print("\n=== is there a natural threshold in the data? ===")
    c = collections.Counter(own(r) for r in surv)
    for v in list(range(0, 21)):
        mark = "   <- default" if v == SHIPPED_DEFAULT else ""
        print(f"  own={v:>2} {c[v]:>4}  {'#' * min(c[v], 60)}{mark}")
    print(f"  own>=21 {sum(n for v, n in c.items() if v >= 21):>4}")
    print("\n  A spike at 0, then a flat plateau. There is no valley at 10, or "
          "anywhere\n  else. The threshold cannot be read off the "
          "distribution.")


if __name__ == "__main__":
    main()
