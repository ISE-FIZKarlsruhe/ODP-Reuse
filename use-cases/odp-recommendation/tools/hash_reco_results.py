#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hash_reco_results.py -- produce the hashed-identifier variant of the ODP
recommendation results.

WHY THIS EXISTS
---------------
The ODP-Reuse dataset is released in two forms: a plain form (real ontology
names and IRIs) and a hashed form in which every IRI is replaced by
SHA-1 + URL-safe-base64 (see ``MultiSource/hash_iris.py``).  ``hash_iris.py``
rewrites IRIs *inside* RDF files but deliberately preserves directory names and
file names, so a results file keyed by ontology directory name and pattern stem
still carries the original, identifying names.

This script closes that gap for the recommendation results: it rewrites the
ontology identifiers and the recommended/ground-truth pattern stems with the
SAME hash function used by ``hash_iris.py``, so the hashed results can be
consumed alongside the hashed dataset release without re-exposing the plain
names.  The crosswalk is written to ``id_map.csv``.

The hash is deterministic and identical to ``hash_iris.get_hashed_iri`` minus
the IRI prefix, so a consumer who has both releases can join them.

Metric values are copied through untouched -- only identifiers change.

Usage
-----
    python tools/hash_reco_results.py \
        --in-dir  results/noOBO \
        --out-dir results_hashed/noOBO
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from pathlib import Path

# Identifier fields in the per-ontology result rows that name an ontology.
ONTOLOGY_FIELDS = ("ontology",)
# Fields holding ranked lists / sets of pattern stems.
PATTERN_LIST_FIELDS = ("top", "truth", "recommended", "ground_truth")


def hash_id(value: str) -> str:
    """SHA-1 + URL-safe base64 (no padding) -- same function as hash_iris.py."""
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


class Hasher:
    """Hashes identifiers and records the crosswalk."""

    def __init__(self) -> None:
        self.mapping: dict[tuple[str, str], str] = {}

    def __call__(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key not in self.mapping:
            self.mapping[key] = hash_id(value)
        return self.mapping[key]

    def write_map(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["kind", "original", "hashed"])
            for (kind, original), hashed in sorted(self.mapping.items()):
                writer.writerow([kind, original, hashed])


def hash_row(row: dict, hasher: Hasher) -> dict:
    out = dict(row)
    for field in ONTOLOGY_FIELDS:
        if isinstance(out.get(field), str):
            out[field] = hasher("ontology", out[field])
    for field in PATTERN_LIST_FIELDS:
        value = out.get(field)
        if isinstance(value, list):
            out[field] = [
                hasher("pattern", item) if isinstance(item, str) else item
                for item in value
            ]
        elif isinstance(value, str):
            out[field] = hasher("pattern", value)
    return out


def hash_result_file(src: Path, dst: Path, hasher: Hasher) -> int:
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = data.get("per_ontology")
    if isinstance(rows, list):
        data["per_ontology"] = [
            hash_row(r, hasher) if isinstance(r, dict) else r for r in rows
        ]
    data["_identifiers"] = "hashed (sha1+urlsafe-b64, see id_map.csv)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return len(rows) if isinstance(rows, list) else 0


def sanitise_name(name: str) -> str:
    """Backend file names embed a model id in brackets; keep them stable."""
    return re.sub(r"[^\w.\[\]@:-]+", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="directory of plain results (e.g. results/noOBO)")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="destination for hashed results")
    args = ap.parse_args()

    hasher = Hasher()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_rows = 0
    for src in sorted(args.in_dir.glob("*.json")):
        dst = args.out_dir / sanitise_name(src.name)
        n_rows += hash_result_file(src, dst, hasher)
        n_files += 1
        print(f"[OK] {src.name} -> {dst.name}")

    # comparison.csv holds only aggregate metrics and method names -- no
    # ontology or pattern identifiers -- so it is copied verbatim.
    comparison = args.in_dir / "comparison.csv"
    if comparison.exists():
        (args.out_dir / "comparison.csv").write_text(
            comparison.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print("[OK] comparison.csv copied verbatim (contains no identifiers)")

    hasher.write_map(args.out_dir / "id_map.csv")
    n_ont = sum(1 for k in hasher.mapping if k[0] == "ontology")
    n_pat = sum(1 for k in hasher.mapping if k[0] == "pattern")
    print(f"\n{n_files} files, {n_rows} per-ontology rows")
    print(f"id_map.csv: {n_ont} ontology ids, {n_pat} pattern stems")


if __name__ == "__main__":
    main()
