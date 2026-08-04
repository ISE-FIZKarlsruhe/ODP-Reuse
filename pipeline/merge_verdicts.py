#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_verdicts.py -- fold per-rater verdict files into `machine_verdicts.csv`.

`adjudicate_labels.py judge` writes one file per invocation, so re-running a
single rater to fill in its missing items produces a side file
(`machine_verdicts_<rater>.csv`) that the merged table does not yet know about.
This script overlays those side files onto the merged one and recomputes
`majority` / `n_agree` with exactly the rule `cmd_judge` uses.

It only ever ADDS or REPLACES a rater's verdict where the side file has one; it
never blanks a verdict the merged file already holds. The previous merged file
is kept as `machine_verdicts.<n>.bak.csv` rather than overwritten, for the same
reason the repair log is: a provenance record that can be overwritten will be.

    python pipeline/merge_verdicts.py --out pipeline/out/adjudication
    python pipeline/adjudicate_labels.py score --out pipeline/out/adjudication
"""
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

RATERS = ["gpt", "gemini", "claude", "deepseek"]
FIELDS = ["verdict", "confidence", "reason"]


def load(p: Path) -> dict[str, dict]:
    if not p.is_file():
        return {}
    return {r["id"]: r for r in csv.DictReader(p.open(encoding="utf-8"))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_p = args.out / "machine_verdicts.csv"
    base = load(base_p)
    if not base:
        raise SystemExit(f"no {base_p}")

    # every side file that carries a rater column
    sides: list[tuple[Path, list[str]]] = []
    for p in sorted(args.out.glob("machine_verdicts_*.csv")):
        rows = load(p)
        if not rows:
            continue
        cols = next(iter(rows.values())).keys()
        present = [r for r in RATERS if f"{r}_verdict" in cols]
        if present:
            sides.append((p, present))

    changes = Counter()
    for p, raters in sides:
        rows = load(p)
        for i, r in rows.items():
            if i not in base:
                continue
            for rater in raters:
                v = (r.get(f"{rater}_verdict") or "").strip()
                if not v:
                    continue
                old = (base[i].get(f"{rater}_verdict") or "").strip()
                if old == v:
                    continue
                changes[(p.name, rater, "filled" if not old else "changed")] += 1
                for f in FIELDS:
                    base[i][f"{rater}_{f}"] = r.get(f"{rater}_{f}", "")

    # recompute majority / n_agree exactly as cmd_judge does
    for i, r in base.items():
        votes = [(r.get(f"{k}_verdict") or "").strip() for k in RATERS]
        votes = [v for v in votes if v]
        if not votes:
            r["majority"], r["n_agree"] = "", ""
            continue
        c = Counter(votes).most_common()
        r["majority"] = c[0][0] if (len(c) == 1 or c[0][1] > c[1][1]) else "unclear"
        r["n_agree"] = c[0][1]

    print("=== overlay ===")
    for (fname, rater, kind), n in sorted(changes.items()):
        print(f"  {fname:34s} {rater:9s} {kind:8s} {n}")
    if not changes:
        print("  nothing to do -- the merged file is already current")

    print("\n=== per-rater coverage after merge ===")
    for rater in RATERS:
        vals = [(r.get(f"{rater}_verdict") or "").strip() for r in base.values()]
        have = [v for v in vals if v]
        c = Counter(have)
        print(f"  {rater:9s} n={len(have):3d}  missing={len(vals) - len(have):2d}  "
              + ", ".join(f"{k} {c[k]}" for k in
                          ("not_reuse", "reuse", "unclear") if c[k]))

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return

    n = 1
    bak = args.out / f"machine_verdicts.{n:02d}.bak.csv"
    while bak.exists():
        n += 1
        bak = args.out / f"machine_verdicts.{n:02d}.bak.csv"
    shutil.copyfile(base_p, bak)
    print(f"\n[OK] previous merged table kept as {bak.name}")

    cols = ["id"] + [f"{r}_{f}" for r in RATERS for f in FIELDS] \
        + ["majority", "n_agree"]
    with base_p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for i in base:
            w.writerow(base[i])
    print(f"[OK] {base_p}  ({len(base)} rows)")
    print("\nNow re-run:  python pipeline/adjudicate_labels.py score "
          f"--out {args.out}")


if __name__ == "__main__":
    main()
