#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_filters.py -- write one reviewable removal report per filter, and
independently RE-VERIFY each removal against the source files.

Every drop the gate makes is a claim about a file. This script re-checks those
claims from scratch -- it does not trust MANIFEST.csv -- so that a human can read
`removed_G2_pattern_copy.csv` and judge whether each exclusion was justified.

For every filter it writes `audit/removed_<rule>.csv` containing one row per
removed ontology with the evidence needed to judge it, plus a `verified` column:

    verified = ok        the removal was re-confirmed from the files
    verified = DISPUTED  re-checking did NOT reproduce the removal -- inspect
    verified = unchecked the check was skipped (see --sample / --skip-slow)

Usage
-----
    python audit_filters.py --manifest out/MANIFEST.csv \\
        --data-root /path/to/MultiSource --out out/audit

    # verify only N per rule (the expensive rules re-parse files)
    python audit_filters.py ... --sample 40
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.getLogger("rdflib").setLevel(logging.ERROR)

from build_manifest import DATASETS  # noqa: E402
from odpqual.entities import (  # noqa: E402
    collect_entities,
    declared_ontology_iris,
    parse_graph,
    strip_annotations,
)
from odpqual.gate import (  # noqa: E402
    in_pattern_library,
    main_ontology_file,
    pattern_files,
    sha256,
)


def onto_dir(data_root: Path, r: dict) -> Path:
    return data_root / DATASETS[r["dataset"]] / r["ontology_id"]


def load(g_path: Path, strip: bool = True):
    g, err = parse_graph(g_path)
    if g is None:
        return None, err
    if strip:
        g, _ = strip_annotations(g)
    return g, ""


# ----------------------------------------------------------------------
# per-rule verification: each returns (verified, evidence_dict)
# ----------------------------------------------------------------------

def verify_g1(d: Path, r: dict) -> tuple[str, dict]:
    """Re-attempt the parse. The removal is right iff it still fails."""
    main = main_ontology_file(d)
    if main is None:
        return "ok", {"evidence": "no RDF file in directory",
                      "file": "", "size_mb": ""}
    g, err = parse_graph(main)
    size = f"{main.stat().st_size / 1e6:.2f}"
    if g is None:
        return "ok", {"evidence": err, "file": main.name, "size_mb": size}
    return "DISPUTED", {"evidence": f"re-parsed OK ({len(g)} triples)",
                        "file": main.name, "size_mb": size}


def verify_g2(d: Path, r: dict) -> tuple[str, dict]:
    """Re-check all three pattern-copy tests independently."""
    main = main_ontology_file(d)
    if main is None:
        return "DISPUTED", {"evidence": "no ontology file"}
    g, err = load(main)
    if g is None:
        return "DISPUTED", {"evidence": f"does not parse: {err}"}
    iris = declared_ontology_iris(g)
    hits = [i for i in iris if in_pattern_library(i)]
    if hits:
        return "ok", {"test": "declared_iri", "evidence": hits[0],
                      "declared_iris": ";".join(iris[:3])}
    mh = sha256(main)
    for pf in pattern_files(d):
        if sha256(pf) == mh:
            return "ok", {"test": "identical_file", "evidence": pf.name,
                          "declared_iris": ";".join(iris[:3])}
    ents = set(collect_entities(g, wide=True))
    for pf in pattern_files(d):
        pg, _ = load(pf)
        if pg is None:
            continue
        pe = set(collect_entities(pg, wide=True))
        if pe and pe == ents:
            return "ok", {"test": "identical_entities",
                          "evidence": f"{pf.stem} ({len(pe)} entities)",
                          "declared_iris": ";".join(iris[:3])}
    return "DISPUTED", {"test": "", "evidence": "no pattern-copy test fires",
                        "declared_iris": ";".join(iris[:3])}


def verify_g3(d: Path, r: dict, keeper_dir: Path | None) -> tuple[str, dict]:
    """A duplicate is right iff the two files really are byte-identical."""
    main = main_ontology_file(d)
    if main is None or keeper_dir is None:
        return "DISPUTED", {"evidence": "keeper or file missing"}
    kmain = main_ontology_file(keeper_dir)
    if kmain is None:
        return "DISPUTED", {"evidence": "keeper has no ontology file"}
    h1, h2 = sha256(main), sha256(kmain)
    if h1 == h2:
        return "ok", {"evidence": f"sha256 match {h1[:16]}...",
                      "kept_instead": keeper_dir.name,
                      "size_mb": f"{main.stat().st_size / 1e6:.2f}"}
    return "DISPUTED", {"evidence": f"hashes DIFFER {h1[:12]} vs {h2[:12]}",
                        "kept_instead": keeper_dir.name, "size_mb": ""}


def verify_g4(d: Path, r: dict, threshold: int) -> tuple[str, dict]:
    """Re-count entities the ontology contributes beyond its patterns."""
    main = main_ontology_file(d)
    if main is None:
        return "DISPUTED", {"evidence": "no ontology file"}
    g, err = load(main)
    if g is None:
        return "DISPUTED", {"evidence": f"does not parse: {err}"}
    ents = set(collect_entities(g, wide=True))
    supplied: set[str] = set()
    for pf in pattern_files(d):
        pg, _ = load(pf)
        if pg is not None:
            supplied |= set(collect_entities(pg, wide=True))
    own = len(ents - supplied)
    ok = own < threshold
    return ("ok" if ok else "DISPUTED"), {
        "evidence": f"own_entities={own} (threshold {threshold})",
        "n_entities_wide": len(ents), "n_triples": len(g),
        "n_pattern_files": len(pattern_files(d))}


def verify_g5(d: Path, r: dict) -> tuple[str, dict]:
    """No-ground-truth is right iff no pattern shares an entity with the host."""
    main = main_ontology_file(d)
    if main is None:
        return "DISPUTED", {"evidence": "no ontology file"}
    g, err = load(main)
    if g is None:
        return "DISPUTED", {"evidence": f"does not parse: {err}"}
    ents = set(collect_entities(g, wide=True))
    best, best_p = 0, ""
    n_pat = 0
    for pf in pattern_files(d):
        pg, _ = load(pf)
        if pg is None:
            continue
        n_pat += 1
        inter = len(ents & set(collect_entities(pg, wide=True)))
        if inter > best:
            best, best_p = inter, pf.stem
    if best == 0:
        return "ok", {"evidence": f"0 overlap across {n_pat} pattern file(s)",
                      "n_entities_wide": len(ents), "n_pattern_files": n_pat}
    return "DISPUTED", {"evidence": f"overlap {best} with {best_p}",
                        "n_entities_wide": len(ents), "n_pattern_files": n_pat}


# ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-own-entities", type=int, default=1,
                    help="must match the threshold the manifest was built "
                         "with; 1 is the released value")
    ap.add_argument("--sample", type=int, default=0,
                    help="verify at most N removals per rule (0 = all)")
    ap.add_argument("--rules", nargs="*",
                    default=["G1_parse", "G2_pattern_copy", "G3_duplicate",
                             "G4_substantive", "G5_has_gt"])
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    by_id = {(r["dataset"], r["ontology_id"]): r for r in rows}

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["drop_reason"]:
            grouped[r["drop_reason"].split(":", 1)[0]].append(r)

    summary = []
    for rule in args.rules:
        removed = grouped.get(rule, [])
        if not removed:
            print(f"[{rule}] nothing removed")
            continue
        to_check = removed if not args.sample else removed[:args.sample]
        print(f"[{rule}] {len(removed)} removed, verifying {len(to_check)} ...")

        out_rows = []
        verdicts = Counter()
        for i, r in enumerate(removed, 1):
            d = onto_dir(args.data_root, r)
            base = {
                "dataset": r["dataset"], "ontology_id": r["ontology_id"],
                "drop_reason": r["drop_reason"],
                "n_entities_wide": r["n_entities_wide"],
                "n_pattern_files": r["n_pattern_files"],
                "n_own_entities": r["n_own_entities"],
                "path": r["path"],
            }
            if r not in to_check or not d.is_dir():
                base.update({"verified": "unchecked", "evidence": ""})
                verdicts["unchecked"] += 1
                out_rows.append(base)
                continue
            try:
                if rule == "G1_parse":
                    v, ev = verify_g1(d, r)
                elif rule == "G2_pattern_copy":
                    v, ev = verify_g2(d, r)
                elif rule == "G3_duplicate":
                    keeper = r["duplicate_of"]
                    kd = onto_dir(args.data_root,
                                  by_id.get((r["dataset"], keeper), r)) \
                        if keeper else None
                    v, ev = verify_g3(d, r, kd)
                elif rule == "G4_substantive":
                    v, ev = verify_g4(d, r, args.min_own_entities)
                elif rule == "G5_has_gt":
                    v, ev = verify_g5(d, r)
                else:
                    v, ev = "unchecked", {}
            except Exception as exc:  # noqa: BLE001
                v, ev = "DISPUTED", {"evidence": f"audit error: "
                                                 f"{type(exc).__name__}: {exc}"}
            base.update({"verified": v, **ev})
            verdicts[v] += 1
            out_rows.append(base)
            if i % 50 == 0:
                print(f"    {i}/{len(removed)}", flush=True)

        cols = sorted({k for row in out_rows for k in row},
                      key=lambda c: (c not in ("dataset", "ontology_id",
                                               "verified", "evidence"), c))
        p = args.out / f"removed_{rule}.csv"
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows)
        print(f"    -> {p.name}  {dict(verdicts)}")
        summary.append({"rule": rule, "n_removed": len(removed),
                        **{k: v for k, v in verdicts.items()}})

    if summary:
        cols = sorted({k for r in summary for k in r},
                      key=lambda c: (c not in ("rule", "n_removed"), c))
        with (args.out / "audit_summary.csv").open("w", newline="",
                                                   encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        print("\n=== AUDIT SUMMARY ===")
        for s in summary:
            dis = s.get("DISPUTED", 0)
            flag = f"  <-- {dis} DISPUTED, INSPECT" if dis else ""
            print(f"  {s['rule']:20s} removed={s['n_removed']:>5d} "
                  f"ok={s.get('ok', 0):>5d} disputed={dis:>4d} "
                  f"unchecked={s.get('unchecked', 0):>5d}{flag}")


if __name__ == "__main__":
    main()
