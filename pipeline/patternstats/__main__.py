"""Run all four pattern-library analyses.

    python -m pipeline.patternstats \
        --data-root "C:/Users/eno/Documents/PhD/MultiSource_server" \
        --out pipeline/out/patternstats

The libraries are parsed once into ``<out>/cache/`` and reused; re-running is
cheap.  ``--only`` restricts to a subset of analyses.  ``RUN.json`` in the output
directory records the arguments, the file inventory and the headline counts, so
any CSV in the directory can be traced back to the run that made it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from . import domains, extract, libraries, overlap, shape, toplevel, units

ANALYSES = ("shape", "toplevel", "overlap", "domains")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path,
                    help="directory holding patterns-repository/, manchester/, "
                         "dos-dp/, Industrial-Standard-Ontology-Design-Patterns/")
    ap.add_argument("--out", type=Path, default=Path("pipeline/out/patternstats"),
                    help="output directory for the CSVs")
    ap.add_argument("--cache", type=Path, default=None,
                    help="parse cache (default <out>/cache)")
    ap.add_argument("--pairs", type=Path,
                    default=Path("pipeline/out/rq_iri_gated/pairs.csv"),
                    help="reuse evidence, for the domain x reuse cross-tabulation")
    ap.add_argument("--unit", default=units.DEFAULT_UNIT, choices=list(units.UNITS),
                    help=f"unit of analysis (default {units.DEFAULT_UNIT})")
    ap.add_argument("--jaccard-unit", default="content", choices=list(units.UNITS),
                    help="unit for the entity-set Jaccard pass (default content)")
    ap.add_argument("--reuse-unit", default="stem_global", choices=list(units.UNITS),
                    help="unit for the domain x reuse cross-tabulation (default "
                         "stem_global: pairs.csv keys a pattern by stem alone, so "
                         "any other unit either double-counts a stem with several "
                         "representatives or drops one that was deduplicated away)")
    ap.add_argument("--min-ref-iris", type=int, default=1,
                    help="distinct IRIs from a vocabulary needed for a "
                         "reference-based alignment (default 1 = ungated)")
    ap.add_argument("--only", nargs="*", default=list(ANALYSES), choices=list(ANALYSES))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--libs", nargs="*", default=None,
                    choices=list(libraries.LIB_ORDER))
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or (out / "cache")

    t0 = time.time()
    recs_by_key = extract.build(args.data_root, cache, resume=not args.no_resume,
                               libs=args.libs)
    records = sorted(recs_by_key.values(), key=lambda r: (r["library"], r["rel_path"]))
    t_extract = time.time() - t0

    run_info: dict = {
        "data_root": str(args.data_root),
        "out": str(out),
        "unit": args.unit,
        "jaccard_unit": args.jaccard_unit,
        "reuse_unit": args.reuse_unit,
        "min_ref_iris": args.min_ref_iris,
        "pairs_csv": str(args.pairs),
        "n_files": len(records),
        "n_parsed": sum(1 for r in records if r.get("parsed")),
        "extract_seconds": round(t_extract, 1),
        "analyses": list(args.only),
    }

    if "shape" in args.only:
        t = time.time()
        res = shape.run(records, out, unit=args.unit)
        run_info["shape"] = {"inventory": res["inventory"],
                             "seconds": round(time.time() - t, 1)}
        print(f"[shape] {len(res['per_pattern'])} patterns at unit={args.unit}")

    if "toplevel" in args.only:
        t = time.time()
        res = toplevel.run(records, out, unit=args.unit,
                           min_ref_iris=args.min_ref_iris)
        run_info["toplevel"] = {"summary": res["summary"],
                                "seconds": round(time.time() - t, 1)}
        print(f"[toplevel] {len(res['per_pattern'])} patterns, "
              f"{len(res['by_library'])} library x vocabulary rows")

    if "overlap" in args.only:
        t = time.time()
        res = overlap.run(records, cache, out, unit=args.unit,
                          jaccard_unit=args.jaccard_unit)
        run_info["overlap"] = {
            "n_hash_groups_multi": len(res["hash_groups"]),
            "n_stem_groups_multi": len(res["stem_groups"]),
            "n_cross_library_stems": len(res["cross_library_stems"]),
            "n_jaccard_pairs_gt0": len(res["jaccard_pairs"]),
            "n_jaccard_patterns": res["n_jaccard_patterns"],
            "n_distinct_entity_iris": res["n_distinct_entity_iris"],
            "n_terms_spanning_libraries": res["n_terms_spanning_libraries"],
            "cross_refs": res["cross_refs"],
            "seconds": round(time.time() - t, 1),
        }
        print(f"[overlap] {len(res['hash_groups'])} multi-file hash groups, "
              f"{len(res['cross_library_stems'])} cross-library stems, "
              f"{len(res['jaccard_pairs'])} pairs with J>0")

    if "domains" in args.only:
        t = time.time()
        res = domains.run(records, out, pairs_csv=args.pairs, unit=args.unit,
                          reuse_unit=args.reuse_unit)
        run_info["domains"] = {
            "uncertainty": res["uncertainty"],
            "reuse_join": res.get("reuse_join", {}),
            "seconds": round(time.time() - t, 1),
        }
        print(f"[domains] {len(res['per_pattern'])} patterns classified")

    (out / "RUN.json").write_text(json.dumps(run_info, indent=2, default=str),
                                  encoding="utf-8")
    print(f"[done] {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
