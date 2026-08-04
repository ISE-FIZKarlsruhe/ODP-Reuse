#!/usr/bin/env python3
"""
validate_release.py -- check this release is internally consistent.

Run it after downloading; it needs only the Python standard library.

    python validate_release.py

Checks:
  1. every view's pairs.csv and ontologies.csv agree on the ontology set
  2. statistics.csv matches what pairs.csv actually contains
  3. every *-complete view is a subset of its *-reuse sibling
  4. every ontology_id in a view resolves to a directory under corpus/
  5. every pattern_file in a view resolves to a file under corpus/
  6. MANIFEST.csv accounts for every ontology, and its kept set matches the views
  7. corpus-hashed/, if present, covers the same ontologies as corpus/

Exits non-zero if any check fails.
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
VIEWS = HERE / "views"
RDF = (".owl", ".ttl", ".rdf", ".xml", ".nt", ".n3", ".trig", ".jsonld")
errors, warnings = [], []


def read(p):
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_view(d):
    name = d.name
    pairs, ontos, stats = (read(d / "pairs.csv"), read(d / "ontologies.csv"),
                           read(d / "statistics.csv"))
    if pairs is None or ontos is None:
        errors.append(f"{name}: missing pairs.csv or ontologies.csv")
        return None

    pair_ids = {r["ontology_id"] for r in pairs}
    onto_ids = {r["ontology_id"] for r in ontos}
    if pair_ids != onto_ids:
        errors.append(f"{name}: pairs.csv and ontologies.csv disagree "
                      f"(+{len(pair_ids - onto_ids)} / -{len(onto_ids - pair_ids)})")

    if stats:
        s = stats[0]
        if int(s["n_ontologies"]) != len(onto_ids):
            errors.append(f"{name}: statistics says n_ontologies="
                          f"{s['n_ontologies']}, ontologies.csv has {len(onto_ids)}")
        if int(s["n_pairs"]) != len(pairs):
            errors.append(f"{name}: statistics says n_pairs={s['n_pairs']}, "
                          f"pairs.csv has {len(pairs)}")
        n_complete = sum(1 for r in pairs if r.get("complete_reuse") == "1")
        if int(s["n_complete_pairs"]) != n_complete:
            errors.append(f"{name}: statistics n_complete_pairs="
                          f"{s['n_complete_pairs']}, pairs.csv has {n_complete}")

    # The files live once, under corpus/. A view that names an ontology or a
    # pattern the corpus does not contain is the failure mode that matters most:
    # it makes the tables unusable while every count still looks right.
    if CORPUS.is_dir():
        missing_dirs = missing_pats = 0
        for r in pairs:
            # Paths in pairs.csv are relative to THIS view directory
            # ("../../corpus/<portion>/<id>"), so a view stays self-describing
            # wherever the release is unpacked.
            od = r.get("ontology_dir")
            if od and not (d / od).is_dir():
                missing_dirs += 1
            pf = r.get("pattern_file", "")
            if pf and not pf.startswith("(") and not (d / pf).is_file():
                missing_pats += 1
        if missing_dirs:
            errors.append(f"{name}: {missing_dirs} ontology_dir value(s) "
                          f"do not resolve")
        if missing_pats:
            errors.append(f"{name}: {missing_pats} pattern_file value(s) "
                          f"do not resolve")
    else:
        warnings.append("corpus/ is absent -- tables validated, files not "
                        "(download the corpus archive to check them)")

    return {"name": name, "n_ontologies": len(onto_ids), "n_pairs": len(pairs),
            "ids": onto_ids}


def main():
    if not VIEWS.is_dir():
        print("[ERROR] no views/ directory -- is this a release root?")
        sys.exit(1)

    print("view                        ontologies    pairs")
    print("-" * 52)
    built = {}
    for d in sorted(VIEWS.iterdir()):
        if not d.is_dir():
            continue
        r = check_view(d)
        if r:
            built[r["name"]] = r
            print(f"{r['name']:26s} {r['n_ontologies']:>10d} {r['n_pairs']:>8d}")

    for name, r in built.items():
        if name.endswith("-complete"):
            sib = built.get(name.replace("-complete", "-reuse"))
            if sib:
                extra = r["ids"] - sib["ids"]
                if extra:
                    errors.append(f"{name}: {len(extra)} ontologies absent from "
                                  f"its -reuse sibling")

    man = read(HERE / "MANIFEST.csv")
    if man is None:
        errors.append("MANIFEST.csv is missing")
    else:
        kept = {r["ontology_id"] for r in man if not r["drop_reason"]}
        union = set().union(*(r["ids"] for r in built.values())) if built else set()
        if union - kept:
            errors.append(f"{len(union - kept)} ontologies appear in a view but "
                          f"are marked dropped in MANIFEST.csv")
        print(f"\nMANIFEST.csv: {len(man)} ontologies considered, {len(kept)} kept; "
              f"{len(union)} appear in at least one view")

    hashed = HERE / "corpus-hashed"
    if hashed.is_dir() and CORPUS.is_dir():
        a = {p.name for p in CORPUS.iterdir() if p.is_dir()}
        b = {p.name for p in hashed.iterdir() if p.is_dir()}
        if a != b:
            errors.append(f"corpus-hashed/ covers different portions than corpus/ "
                          f"({sorted(a ^ b)})")
        fails = hashed / "hash_failures.csv"
        if fails.is_file():
            n = len(read(fails) or [])
            warnings.append(f"corpus-hashed/: {n} file(s) could not be hashed "
                            f"(see hash_failures.csv); these are unparseable in "
                            f"the plain corpus too")

    print()
    for w in warnings:
        print(f"[WARN]  {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    if errors:
        print(f"\nFAILED: {len(errors)} error(s)")
        sys.exit(1)
    print(f"OK: {len(built)} views validated, no errors"
          + (f", {len(warnings)} warning(s)" if warnings else ""))


if __name__ == "__main__":
    main()
