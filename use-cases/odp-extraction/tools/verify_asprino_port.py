#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_asprino_port.py -- does the vendored step 1 still do what its authors' code did?

``third_party/conceptual_components/`` runs the authors' 2021 code under Python
3.11 / rdflib 7 / networkx 3 instead of Python 3.7 / rdflib 4.2.2 / networkx 2.4.
That is a claim, and this script is the evidence for it.

The upstream repository ships the communities it published for both of its own
corpora (``conceptual-components-extraction/results/{CH,Conf}/communities_original_uris/``,
one ``*_original_uris.txt`` per community per ontology). This script re-runs the
vendored code over the matching corpus and compares the community count per
ontology against those files.

An exact match is not expected and is not the bar: greedy modularity
maximisation breaks ties by iteration order, and that order changed between
networkx 2.4 and 3.x. The bar is that the counts agree per ontology to within a
small margin and that the totals agree, which is what a faithful port of a
tie-break-sensitive algorithm looks like.

Usage
-----
    python tools/verify_asprino_port.py --repo /path/to/conceptual-components \
        --corpus Conf

    # clone first:
    #   git -c core.longpaths=true clone \
    #       https://github.com/stlab-istc-cnr/conceptual-components.git
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_USECASE))

from rdflib import Graph  # noqa: E402

from extractbench.asprino import communities  # noqa: E402

CORPUS_DIR = {"Conf": "corpora/Conf/Conf_dataset_inferred",
              "CH": "corpora/CH/CH_dataset_inferred"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path,
                    help="clone of stlab-istc-cnr/conceptual-components")
    ap.add_argument("--corpus", choices=("Conf", "CH"), default="Conf")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = args.repo / "conceptual-components-extraction"
    src = root / CORPUS_DIR[args.corpus]
    pub = root / "results" / args.corpus / "communities_original_uris"
    if not src.is_dir() or not pub.is_dir():
        sys.exit(f"not found: {src} or {pub}")

    rows = []
    print(f"{'ontology':32s} {'ours':>6s} {'published':>10s} {'diff':>5s} {'sec':>7s}")
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        g = Graph()
        try:
            g.parse(f.as_posix(), format="turtle")
        except Exception as exc:  # noqa: BLE001
            print(f"{f.stem:32s} parse failed: {type(exc).__name__}")
            continue
        t = time.time()
        ours = len(communities(g))
        el = time.time() - t
        pdir = pub / f.stem
        theirs = len(list(pdir.glob("*_original_uris.txt"))) if pdir.is_dir() else 0
        rows.append({"ontology": f.stem, "ours": ours, "published": theirs,
                     "diff": ours - theirs, "seconds": round(el, 2)})
        print(f"{f.stem:32s} {ours:6d} {theirs:10d} {ours - theirs:+5d} {el:7.2f}",
              flush=True)

    to = sum(r["ours"] for r in rows)
    tp = sum(r["published"] for r in rows)
    exact = sum(1 for r in rows if r["diff"] == 0)
    print(f"\n{'TOTAL':32s} {to:6d} {tp:10d} {to - tp:+5d}")
    print(f"{len(rows)} ontologies, {exact} exact, "
          f"total difference {100 * abs(to - tp) / max(tp, 1):.1f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"[OK] {args.out}")


if __name__ == "__main__":
    main()
