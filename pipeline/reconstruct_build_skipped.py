#!/usr/bin/env python3
"""reconstruct_build_skipped.py -- rebuild the adjudication skip log.

Why this exists
---------------
`adjudicate_labels.py build` collected its skipped ontologies into a list and
then discarded it: the CLI help promised `build_skipped.csv` and no code path
wrote one. The defect is now fixed in `adjudicate_labels.py`, so any future
build writes the file directly -- but re-running the build would take hours
(it parses every ontology and every pattern), and the adjudication that the
paper reports was performed against the item set as it already exists.

This script therefore reconstructs the log without re-parsing anything,
which is possible because **both** skip criteria are decidable from the
manifest plus the file system:

  1. `--max-mb 40`        -- the ontology FILE is larger than 40 MB.
                             Checked from the file size on disk.
  2. `--max-entities 20000` -- the ontology has more than 20,000 entities
                             under the WIDE definition. That count is the
                             manifest column `n_entities_wide`.

and because the candidate population is likewise a manifest filter: rows whose
`drop_reason` does not start with `G1_parse`, restricted to the `github` and
`noOBO` datasets (DOS-DP is excluded from the item set by default).

Rule 1 is evaluated before rule 2 in `cmd_build`, so it takes precedence here
too, and each ontology is attributed to the FIRST rule it trips -- the same
convention the quality gate uses.

Output: `<out>/build_skipped.csv`, columns ontology,dataset,reason,detail --
identical in shape to what the fixed `cmd_build` now writes, plus a
`reconstructed` column so the file can never be mistaken for a live trace.

Usage
-----
    python pipeline/reconstruct_build_skipped.py \
        --manifest pipeline/out/MANIFEST.csv \
        --data-root <corpus> \
        --out pipeline/out/adjudication
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

# same mapping cmd_build uses
DATASETS = {"github": "github_dataset", "noOBO": "noOBO", "OBO": "OBO"}
ONTO_EXT = {".owl", ".ttl", ".rdf", ".xml", ".nt", ".n3", ".trig", ".jsonld"}


def main_ontology_file(d: Path) -> Path | None:
    """Largest RDF-looking file not under a patterns/ directory."""
    if not d.is_dir():
        return None
    cands = [p for p in d.rglob("*")
             if p.is_file() and p.suffix.lower() in ONTO_EXT
             and "patterns" not in {q.name.lower() for q in p.parents}]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--datasets", nargs="+", default=["github", "noOBO"])
    ap.add_argument("--max-mb", type=float, default=40.0)
    ap.add_argument("--max-entities", type=int, default=20000)
    a = ap.parse_args()

    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh)
                if not r["drop_reason"].startswith("G1_parse")
                and r["dataset"] in a.datasets]

    skipped, no_file = [], 0
    for r in rows:
        d = a.data_root / DATASETS[r["dataset"]] / r["ontology_id"]
        main = main_ontology_file(d)
        if main is None:
            no_file += 1
            skipped.append({"ontology": r["ontology_id"], "dataset": r["dataset"],
                            "reason": "no_ontology_file",
                            "detail": f"no RDF file under {d.name}",
                            "reconstructed": "yes"})
            continue
        mb = main.stat().st_size / 1024 / 1024
        if a.max_mb and mb > a.max_mb:
            skipped.append({"ontology": r["ontology_id"], "dataset": r["dataset"],
                            "reason": "file_too_large",
                            "detail": f"{mb:.1f}MB>{a.max_mb:g}MB",
                            "reconstructed": "yes"})
            continue
        n = int(r["n_entities_wide"] or 0)
        if a.max_entities and n > a.max_entities:
            skipped.append({"ontology": r["ontology_id"], "dataset": r["dataset"],
                            "reason": "too_many_entities",
                            "detail": f"{n}>{a.max_entities}",
                            "reconstructed": "yes"})

    a.out.mkdir(parents=True, exist_ok=True)
    p = a.out / "build_skipped.csv"
    fields = ["ontology", "dataset", "reason", "detail", "reconstructed"]
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(skipped)

    print(f"candidate rows scanned : {len(rows)}")
    print(f"skipped                : {len(skipped)}")
    for k, v in Counter(x["reason"] for x in skipped).most_common():
        print(f"   {k:<20} {v}")
    print(f"[OK] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
