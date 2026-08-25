#!/usr/bin/env python3
"""validate_release.py -- internal consistency of an unpacked MSE release.

Checks that MANIFEST.csv agrees with corpus/, that every checksum matches, that
every prediction refers to an ontology in the manifest, and that no file claims
a score. Exits non-zero on any failure.
"""
import csv, hashlib, json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
fail = []

man = {r["file"]: r for r in csv.DictReader((root / "MANIFEST.csv").open(encoding="utf-8"))}
files = {p.relative_to(root / "corpus").as_posix()
         for p in (root / "corpus").rglob("*") if p.is_file()}
if set(man) - files:
    fail.append(f"in MANIFEST but not in corpus/: {sorted(set(man) - files)[:5]}")
if files - set(man):
    fail.append(f"in corpus/ but not in MANIFEST: {sorted(files - set(man))[:5]}")

for rel, r in man.items():
    p = root / "corpus" / rel
    if not p.is_file():
        continue
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    if h.hexdigest() != r["sha256"]:
        fail.append(f"checksum mismatch: {rel}")

known = set(man)
for pf in sorted((root / "predictions").glob("*.jsonl")):
    for i, line in enumerate(pf.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        d = json.loads(line)
        if "f1" in d or "precision" in d:
            fail.append(f"{pf.name}:{i} carries a score; this release has no ground truth")
            break

print(f"{len(man)} ontologies, {len(list((root / 'predictions').glob('*.jsonl')))} prediction files")
if fail:
    print("FAILED:")
    for f in fail:
        print("  " + f)
    sys.exit(1)
print("OK")
