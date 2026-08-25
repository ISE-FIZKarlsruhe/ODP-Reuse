#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_cc_figures.py -- check a rendered conceptual-components tree, and repair
the records that a transient failure can leave stale.

Three things are checked per ontology directory:

  * every community in ``communities.json`` has the figure it claims, and every
    figure on disk is claimed by a community;
  * every community has its RDF file, if ``cc_to_rdf.py`` has been run;
  * no Graphviz source file was left behind.

The one repair it makes is to the ``figure`` field. ``graphviz.render`` deletes
its ``.gv`` source only after the image is written and lets a failure to delete
propagate, so a file lock held for an instant by a scanner or indexer produced a
perfectly good figure that the run recorded as ``null``. Where the figure is on
disk and the record says otherwise, the record is corrected and the orphaned
source removed.

    python tools/verify_cc_figures.py --root results/cc-figures
    python tools/verify_cc_figures.py --root results/cc-figures --repair
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--ext", default="ttl")
    ap.add_argument("--repair", action="store_true",
                    help="fix recoverable figure records and drop stray sources")
    args = ap.parse_args()
    args.root = args.root.resolve()

    n_onto = n_comm = 0
    fixed = strays = 0
    missing_fig: list[str] = []
    missing_rdf: list[str] = []
    unclaimed: list[str] = []
    by_ds: dict[str, dict[str, int]] = {}

    for js in sorted(args.root.glob("*/*/communities.json")):
        ds = js.parent.parent.name
        if args.datasets and ds not in args.datasets:
            continue
        meta = json.loads(js.read_text(encoding="utf-8"))
        d = js.parent
        n_onto += 1
        stat = by_ds.setdefault(ds, {"ontologies": 0, "communities": 0,
                                     "figures": 0, "rdf": 0})
        stat["ontologies"] += 1
        dirty = False

        on_disk = {p.name for p in d.glob("Community*.png")}
        on_disk |= {p.name for p in d.glob("Community*.svg")}
        on_disk |= {p.name for p in d.glob("Community*.pdf")}
        claimed = set()
        width = max(3, len(str(len(meta["communities"]))))

        for c in meta["communities"]:
            n_comm += 1
            stat["communities"] += 1
            stem = f"Community{c['community']:0{width}d}"

            fig = c.get("figure")
            if not fig:
                # The figure may exist despite the record: look for it.
                found = next((n for n in on_disk if Path(n).stem == stem), None)
                if found:
                    if args.repair:
                        c["figure"] = found
                        dirty = True
                    fixed += 1
                    fig = found
                else:
                    missing_fig.append(f"{ds}/{d.name}/{stem}")
            if fig:
                claimed.add(fig)
                if (d / fig).is_file():
                    stat["figures"] += 1
                else:
                    missing_fig.append(f"{ds}/{d.name}/{fig}")

            rdf = d / f"{stem}.{args.ext}"
            if rdf.is_file():
                stat["rdf"] += 1
            elif (d / "RDF_INDEX.json").is_file():
                missing_rdf.append(f"{ds}/{d.name}/{rdf.name}")

            # Graphviz source left behind by a failed cleanup.
            src = d / stem
            if src.is_file():
                strays += 1
                if args.repair:
                    try:
                        src.unlink()
                    except OSError:
                        pass

        unclaimed += [f"{ds}/{d.name}/{n}" for n in sorted(on_disk - claimed)]
        if dirty:
            js.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    print(f"root: {args.root}")
    print(f"  {n_onto} ontology directories, {n_comm} communities")
    for ds, s in sorted(by_ds.items()):
        print(f"    {ds:<10} {s['ontologies']:>4} ontologies  "
              f"{s['communities']:>6} communities  {s['figures']:>6} figures  "
              f"{s['rdf']:>6} rdf")
    verb = "repaired" if args.repair else "recoverable"
    if fixed:
        print(f"  {fixed} figure records {verb} (image on disk, record empty)")
    if strays:
        print(f"  {strays} Graphviz source files "
              f"{'removed' if args.repair else 'left behind'}")
    for label, items in (("figures missing", missing_fig),
                         ("rdf missing", missing_rdf),
                         ("figures on disk claimed by no community", unclaimed)):
        if items:
            print(f"  {len(items)} {label}:")
            for i in items[:10]:
                print(f"      {i}")
    ok = not missing_fig and not missing_rdf and not unclaimed
    if ok and not strays and not fixed:
        print("  [OK] every community has its figure and its RDF; "
              "nothing stray, nothing to repair")
    elif args.repair and ok:
        print("  [OK] repaired; every community now has its figure and its RDF")


if __name__ == "__main__":
    main()
