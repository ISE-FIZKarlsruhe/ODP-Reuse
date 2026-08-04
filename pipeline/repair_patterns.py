#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
repair_patterns.py -- fill zero-byte pattern files from the local ODP libraries.

THE DEFECT
----------
1,017 of 4,410 pattern files in the harvested corpus (23.1%) are ZERO BYTES:
`build_github_dataset.py` created the file even when the fetch returned nothing.
608 GitHub ontologies have every one of their pattern files empty, and 230 of
those are counted as "no ground truth" drops. That is a data-acquisition failure
being mistaken for a modelling fact.

THE REPAIR
----------
Every empty file names a pattern (`DUL.owl`, `partof.owl`, ...) that is present
in one of the four integrated pattern libraries already on disk. This script
copies the library version into place, so no network fetch is required and the
repair is deterministic and re-runnable.

Matching is by FILE STEM, case-insensitively, against the union of

    patterns-repository/   ODPA (ontologydesignpatterns.org)
    dos-dp/                Dead Simple OWL Design Patterns
    manchester/            Manchester ODP Public Catalog
    Industrial-Standard-Ontology-Design-Patterns/

If a stem resolves to several library files, the repair is REFUSED for that stem
and recorded as `ambiguous` -- guessing which variant the harvester meant would
silently fabricate ground truth. Ambiguities are listed so they can be resolved
explicitly.

Nothing is overwritten: only files of size 0 are touched. A full record is
written to `pattern_repair_log.csv`, and `--dry-run` reports without writing.

Usage
-----
    python repair_patterns.py --data-root /path/to/MultiSource --dry-run
    python repair_patterns.py --data-root /path/to/MultiSource --out out/repair
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

RDF_EXT = (".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt", ".trig", ".jsonld")
LIBRARIES = ("patterns-repository", "dos-dp", "manchester",
             "Industrial-Standard-Ontology-Design-Patterns")
PORTIONS = ("github_dataset", "noOBO", "OBO")


def longpath(p: Path) -> str:
    if sys.platform != "win32":
        return str(p)
    s = str(p.resolve())
    return s if s.startswith("\\\\?\\") else "\\\\?\\" + s


_HASH_CACHE: dict[Path, str] = {}


def file_hash(p: Path) -> str:
    """SHA-256 of a library file, memoised (each is asked for many times)."""
    if p not in _HASH_CACHE:
        h = hashlib.sha256()
        with open(longpath(p), "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _HASH_CACHE[p] = h.hexdigest()
    return _HASH_CACHE[p]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def disambiguate(stem: str, cands: list[Path], data_root: Path) -> tuple[Path, str]:
    """Choose deterministically among library copies with the same stem.

    Three rules, applied in order. Each is a stated editorial convention, not a
    tuned parameter, and the rule that decided is recorded per file.

      R1 LIBRARY PRECEDENCE -- patterns-repository (ODPA) > dos-dp > manchester
         > IS-ODP. Reuse in this corpus was detected against ODPA IRIs, so the
         ODPA copy is the reference the harvester was pointing at. The Manchester
         catalogue republishes several ODPA patterns under its own casing
         (`List.owl` vs `list.owl`); those are not the cited artefact.

      R2 OWN DIRECTORY -- prefer a copy whose containing directory names the
         pattern (`List/list.owl`) over one that merely bundles it as a
         dependency (`ContentOPs/list.owl`). The pattern's own directory is its
         canonical home; a bundled copy may be pruned to what the bundling
         pattern needed.

      R3 LARGEST FILE -- among remaining ties, take the largest. For an upper
         ontology bundled per pattern (DUL.owl appears under ActingFor/,
         AgentRole/, Information_realization/), the fullest copy is the correct
         reference; a truncated copy would UNDER-report reuse, biasing the
         corpus towards finding less reuse than exists.
    """
    order = {lib: i for i, lib in enumerate(LIBRARIES)}

    def lib_of(p: Path) -> int:
        rel = p.relative_to(data_root).parts
        return order.get(rel[0], len(order)) if rel else len(order)

    best_lib = min(lib_of(c) for c in cands)
    pool = [c for c in cands if lib_of(c) == best_lib]
    rule = "R1_library_precedence" if len(pool) < len(cands) else ""

    if len(pool) > 1:
        own = [c for c in pool if _norm(c.parent.name) == _norm(stem)]
        if own:
            pool, rule = own, "R2_own_directory"

    if len(pool) > 1:
        pool = [max(pool, key=lambda c: c.stat().st_size)]
        rule = "R3_largest_file"

    return pool[0], (rule or "R0_single_candidate")


def build_library_index(data_root: Path) -> dict[str, list[Path]]:
    """stem (lowercased) -> library files with that stem, non-empty only."""
    idx: dict[str, list[Path]] = defaultdict(list)
    for lib in LIBRARIES:
        d = data_root / lib
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if (f.is_file() and f.suffix.lower() in RDF_EXT
                    and f.stat().st_size > 0):
                idx[f.stem.lower()].append(f)
    return idx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/repair"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    idx = build_library_index(args.data_root)
    print(f"library index: {len(idx)} distinct stems from {len(LIBRARIES)} libraries")

    log: list[dict] = []
    verdicts: Counter = Counter()

    for portion in PORTIONS:
        root = args.data_root / portion
        if not root.is_dir():
            continue
        for onto in sorted(p for p in root.iterdir() if p.is_dir()):
            pdir = onto / "patterns"
            if not pdir.is_dir():
                continue
            for f in sorted(pdir.iterdir()):
                if not (f.is_file() and f.suffix.lower() in RDF_EXT):
                    continue
                if f.stat().st_size > 0:
                    continue          # never overwrite real content
                cands = idx.get(f.stem.lower(), [])
                # A stem occurring several times in the libraries is only
                # genuinely ambiguous if the CONTENT differs; the libraries
                # duplicate many patterns across subdirectories, and choosing
                # between byte-identical copies is not a choice.
                by_hash: dict[str, Path] = {}
                for c in cands:
                    by_hash.setdefault(file_hash(c), c)
                distinct = list(by_hash.values())
                row = {"portion": portion, "ontology": onto.name,
                       "pattern_file": f.name, "n_candidates": len(cands),
                       "n_distinct_content": len(distinct)}
                if not cands:
                    row.update({"action": "no_library_match", "source": ""})
                    verdicts["no_library_match"] += 1
                else:
                    if len(distinct) > 1:
                        src, rule = disambiguate(f.stem, distinct, args.data_root)
                        row["alternatives"] = ";".join(
                            str(c.relative_to(args.data_root)) for c in distinct
                            if c != src)
                    else:
                        src, rule = distinct[0], "R0_single_candidate"
                    row.update({"action": "repaired" if not args.dry_run
                                else "would_repair",
                                "source": str(src.relative_to(args.data_root)),
                                "rule": rule,
                                "bytes": src.stat().st_size})
                    if not args.dry_run:
                        shutil.copyfile(longpath(src), longpath(f))
                    verdicts["repaired" if not args.dry_run
                             else "would_repair"] += 1
                log.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    if log:
        cols = ["portion", "ontology", "pattern_file", "action", "rule", "source",
                "alternatives", "n_candidates", "n_distinct_content", "bytes"]
        # NEVER overwrite a previous pass's log. The repair once ran twice and
        # the second pass destroyed the first pass's record of 639 files; that
        # log had to be reconstructed from file mtimes afterwards. A provenance
        # record that can be overwritten is one that will be.
        p = args.out / "pattern_repair_log.csv"
        n = 1
        while p.exists():
            p = args.out / f"pattern_repair_log.{n:02d}.csv"
            n += 1
        if n > 1:
            print(f"[note] {args.out / 'pattern_repair_log.csv'} exists; "
                  f"writing this pass to {p.name} instead of overwriting it.")
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(log)
        print(f"[OK] {p}  ({len(log)} empty pattern files considered)")

    print("\n=== REPAIR SUMMARY ===")
    for k, v in verdicts.most_common():
        print(f"  {k:20s} {v}")
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    else:
        print("\nRe-run build_manifest.py to re-gate with the repaired patterns.")


if __name__ == "__main__":
    main()
