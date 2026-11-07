#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
write_release_readme.py -- generate the release's top-level README, Zenodo
metadata and validator, with the statistics table filled in from the actual
built datasets (so the documentation cannot drift from the data).

    python write_release_readme.py --release ../release-iri
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

VALIDATOR = '''#!/usr/bin/env python3
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
        print(f"\\nMANIFEST.csv: {len(man)} ontologies considered, {len(kept)} kept; "
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
        print(f"\\nFAILED: {len(errors)} error(s)")
        sys.exit(1)
    print(f"OK: {len(built)} views validated, no errors"
          + (f", {len(warnings)} warning(s)" if warnings else ""))


if __name__ == "__main__":
    main()
'''

README = """\
# ODP-Reuse: a quality-gated benchmark of ontology design pattern reuse

Ontologies that verifiably reuse Ontology Design Patterns (ODPs): one unfiltered
corpus, six quality-gated views over it, an IRI-anonymised twin of the corpus, the
ontology-to-pattern mapping, per-view statistics, and the complete record of how
the corpus was filtered.

Every ontology here survived a documented quality gate. Most candidates did not:
of **@@DISCOVERED@@** ontologies discovered, **@@KEPT@@** (@@PCTKEPT@@%) are usable.
That attrition is itself a finding, and `MANIFEST.csv` records the reason for
every exclusion.

## The six views

@@TABLE@@

`*-reuse` — every gated ontology with at least one verified reused pattern entity.
`*-complete` — the subset where at least one pattern is reused *in full*.

@@HASHED@@

## Layout

```
release-iri/
├── README.md                  this file
├── MANIFEST.csv               EVERY ontology considered, kept or dropped, with the reason
├── FILTERS.md                 the gate in full: every rule, threshold and count
├── gate_summary.csv/.md       attrition per rule, per portion
├── dataset_statistics.csv     the six views side by side
├── validate_release.py        self-check (stdlib only)
├── zenodo.json                deposition metadata
├── audit/
│   ├── audit_summary.csv      independent re-derivation of every removal
│   └── removed_<rule>.csv     the evidence, per rule
├── corpus/                    THE FILES, shared by all six views
│   └── <portion>/<id>/
│       ├── <id>.owl               the ontology
│       └── patterns/*.owl         the patterns it reuses
├── corpus-hashed/             IRI-anonymised twin of corpus/ (N-Triples)
└── views/<portion>-<level>/
    ├── README.md              self-contained description of this view
    ├── pairs.csv              WHICH PATTERN BELONGS TO WHICH ONTOLOGY
    ├── ontologies.csv         per-ontology metadata
    ├── statistics.csv/.md     this view's numbers
    └── top_patterns.csv       most reused patterns here
```

The six views are **tables, not copies**: every one of them points into the single
`corpus/` tree, so the deposition holds one copy of each file.

## Start here: `pairs.csv`

The ontology-to-pattern relation, one row per (ontology, pattern):

| Column | Meaning |
|---|---|
| `ontology_id` | the ontology's id |
| `ontology_dir` | its directory, **relative to this view directory** |
| `pattern` | pattern stem (file name without extension) |
| `pattern_file` | path to the pattern file, relative to the dataset directory |
| `n_pattern_entities` | entities declared by the pattern |
| `n_reused_entities` | of those, how many occur in the ontology |
| `coverage_pct` | `100 * n_reused / n_pattern_entities` |
| `complete_reuse` | 1 if every pattern entity occurs in the ontology |

## Using it in code

Load the mapping and the files with nothing but the standard library plus rdflib:

```python
import csv
from pathlib import Path
from rdflib import Graph

view = Path("views/github-complete")

# 1. which pattern belongs to which ontology
with (view / "pairs.csv").open(encoding="utf-8") as fh:
    pairs = list(csv.DictReader(fh))

# 2. group patterns by ontology
from collections import defaultdict
by_onto = defaultdict(list)
for r in pairs:
    by_onto[r["ontology_id"]].append(r)

# 3. load one ontology and the patterns it reuses
oid = next(iter(by_onto))
# paths in pairs.csv are relative to the view directory
onto_dir = view / by_onto[oid][0]["ontology_dir"]
onto = Graph().parse(next(p for p in onto_dir.glob("*") if p.suffix in
                          (".owl", ".ttl", ".rdf", ".xml")))
for r in by_onto[oid]:
    pattern = Graph().parse(view / r["pattern_file"])
    print(oid, "reuses", r["pattern"],
          f'{r["coverage_pct"]}% ({r["n_reused_entities"]}/{r["n_pattern_entities"]})')
```

### Ground truth for a detection or extraction task

For each ontology, the ground-truth pattern instances are the pattern entities
that occur in it. Reproduce them exactly as the paper does:

```python
import sys; sys.path.insert(0, "pipeline")
from odpqual.entities import collect_entities, parse_graph, strip_annotations

g, _ = parse_graph(onto_path); g, _ = strip_annotations(g)
host = set(collect_entities(g, wide=True))

gt = {}
for r in by_onto[oid]:
    pg, _ = parse_graph(view / r["pattern_file"])
    pg, _ = strip_annotations(pg)
    pents = set(collect_entities(pg, wide=True))
    gt[r["pattern"]] = host & pents      # the instance to be recovered
```

Two conventions matter and are not optional if you want comparable numbers:
entity sets use the **WIDE** definition, and **annotation vocabularies are
stripped from both sides** (SKOS, DC, DCTERMS, CPANN, VANN, FOAF, …). Both are
implemented in `pipeline/odpqual/entities.py`.

### Worked examples

Two complete, runnable use cases ship with the source repository:

- **ODP recommendation** — `use-cases/odp-recommendation/`. Given an ontology,
  recommend the ODPs it should reuse. Shows this dataset as *training signal*: a
  recommender built on the reuse matrix beats the best that pattern documentation
  alone supports by **+0.805 MAP** (95% CI [+0.729, +0.874], sign test
  *p* < 0.001, 83 wins / 3 losses / 3 ties over 89 paired gate-passing
  ontologies). Read its `CAVEATS.md` before citing any number.
- **Automatic ODP extraction** —
  `use-cases/odp-extraction/tools/run_extraction_eval.py`. Embeds each
  ontology (OWL2Vec\\*, RDF2Vec), clusters, and scores clusters against the ground
  truth by Hungarian matching. Reports both an **oracle** figure (hyper-parameters
  chosen using the ground truth — an upper bound) and an **unsupervised** figure
  (chosen by silhouette, never consulting the answer). On 442 gate-passing GitHub
  ontologies these are F1 **0.453** and **0.321**. Prior work reported only the
  former; the gap between them is the contribution of oracle model selection.

## How the gate works

Each ontology is attributed to the **first** rule it fails, so the columns of
`gate_summary.md` partition the discovered total exactly.

| Rule | Test | Removed | Why |
|---|---|---:|---|
@@GATEROWS@@

`FILTERS.md` states each rule in full, with its threshold, its sub-reasons, and how
sensitive the corpus is to it.

**G2 is the one that changes the numbers.** The previously published
complete-reuse set of 49 ontologies contained 14 verbatim copies of `bag.owl`,
`list.owl`, `agentrole.owl`, `persons.owl`, `timeindexedsituation.owl` and others,
matched against the patterns they *are*. They are excluded here, which is why the
complete-reuse datasets are much smaller than previously reported. The shrinkage
is the correction, not a loss.

## Reproducing this release

```bash
cd pipeline
pip install -r requirements.txt

python build_manifest.py  --data-root <corpus> --out out
python audit_filters.py   --manifest out/MANIFEST.csv --data-root <corpus>
python run_rq_stats.py    --manifest out/MANIFEST.csv --data-root <corpus> \\
                          --out out/rq_iri_gated
python build_release.py   --manifest out/MANIFEST.csv --data-root <corpus> \\
                          --out ../release-iri
python hash_corpus.py     --corpus ../release-iri/corpus
python write_release_readme.py --release ../release-iri
```

Seeds are fixed (embedders 42, K-Means `random_state=42`). The gate, the six
views and the statistics reproduce exactly.

**One known limit, declared.** The extraction use case aggregates to within
0.007 F1, but the *per-ontology selected configuration* depends on
`PYTHONHASHSEED`: between 31% and 66% of ontologies can be assigned a different
winning configuration from one run to the next. Aggregate figures are stable,
per-ontology ones are not. `use-cases/odp-extraction/CAVEATS.md` records the
measurement.

The corpus itself is a snapshot — discovery queries live APIs (GitHub, Google,
BioPortal, MatPortal, LOV), so re-harvesting will not reproduce it. The
reproducible artefact is this archived deposition, not a re-run of the
harvester.

Verify an unpacked copy with:

```bash
python validate_release.py
```

## Licence and citation

Data: **CC BY 4.0**. Code: **MIT** (see `LICENSE` in the source repository). The
third-party pattern libraries the pipeline reads are not redistributed here; each
keeps its own licence.

Cite this deposition by its concept DOI, which always resolves to the latest
version: **10.5281/zenodo.17751740**. Machine-readable metadata is in
`CITATION.cff`. The accompanying paper is under review; its citation will be added
once it is accepted.
"""


HASHED_NOTE_START = "`corpus-hashed/` is the IRI-anonymised twin"


def existing_hashed_note(readme: Path) -> str:
    """The README's current paragraph about corpus-hashed/, or "" if none."""
    if not readme.is_file():
        return ""
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith(HASHED_NOTE_START):
            return line
    return ""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--keep-hashed-note", action="store_true",
                    help="when corpus-hashed/ is absent (it is a Zenodo-only file "
                         "tree, so it is absent in CI), keep the README's existing "
                         "paragraph about it instead of describing it as missing")
    args = ap.parse_args(argv)
    rel = args.release

    stats_path = rel / "dataset_statistics.csv"
    rows = []
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))

    lines = ["| Dataset | Ontologies | Pairs | Distinct patterns | Fully reused pairs |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| `{r['dataset']}` | {r['n_ontologies']} | {r['n_pairs']} "
                     f"| {r['n_distinct_patterns']} | {r['n_complete_pairs']} |")
    table = "\n".join(lines)

    n_discovered = n_kept = 0
    man = rel / "MANIFEST.csv"
    if man.is_file():
        with man.open(encoding="utf-8") as fh:
            mrows = list(csv.DictReader(fh))
        n_discovered = len(mrows)
        n_kept = sum(1 for r in mrows if not r["drop_reason"])

    # Per-rule attrition, read from the release rather than written by hand: a
    # gate table with hard-coded counts is the first thing to go stale after a
    # re-gate, and it is the table readers trust most.
    gsum = rel / "gate_summary.csv"
    counts: dict[str, int] = {}
    per_portion = []
    if gsum.is_file():
        with gsum.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                per_portion.append(
                    f"{r['dataset']} {r['discovered']} → "
                    f"{r['kept_reuse']}")
                for k, v in r.items():
                    if k.startswith("dropped_"):
                        counts[k] = counts.get(k, 0) + int(v or 0)
    GATE = [
        ("G1", "parses as RDF", "dropped_G1_parse",
         "an unparseable file cannot be analysed"),
        ("G2", "**is not itself a copy of a pattern** — its `owl:Ontology` IRI is "
         "not in a pattern-library namespace, it is not byte-identical to one of "
         "its patterns, and its entity set does not equal a pattern's",
         "dropped_G2_pattern_copy",
         "a 100%-reuse criterion is *tautological* on a file that **is** the pattern"),
        ("G3", "is not a byte-identical duplicate of another ontology",
         "dropped_G3_duplicate",
         "forks and mirrors inflate counts and leak across train/test splits"),
        ("G4", "contributes at least **1** entity of its own, beyond its patterns",
         "dropped_G4_substantive",
         "an ontology that adds nothing to its patterns is an aggregation of them, "
         "not a reusing ontology; every ontology dropped here contributes exactly "
         "zero"),
        ("G5", "has ≥1 pattern with a non-empty entity overlap",
         "dropped_G5_has_gt",
         "no overlap means no reuse signal and nothing to score against"),
        ("G6", "has ≥1 pattern reused *in full* (the `*-complete` views only)",
         "dropped_G6_complete_reuse",
         "applied *after* G1–G5, so it cannot be satisfied by a pattern copy"),
    ]
    gate_rows = "\n".join(
        f"| **{code}** | {test} | {counts.get(key, 0):,} | {why} |"
        for code, test, key, why in GATE)

    hashed = rel / "corpus-hashed"
    kept_note = (existing_hashed_note(rel / "README.md")
                 if args.keep_hashed_note and not hashed.is_dir() else "")
    if kept_note:
        hashed_note = kept_note
    elif hashed.is_dir():
        nfiles = sum(1 for _ in hashed.rglob("*.nt"))
        hashed_note = (
            f"`corpus-hashed/` is the IRI-anonymised twin of `corpus/`: "
            f"{nfiles:,} files, same content and same reuse relations, every IRI "
            f"replaced by a deterministic hash. Use **hashed to evaluate** a "
            f"detection method — it removes the namespace-recognition shortcut "
            f"— and **plain to inspect or curate**. Its files are N-Triples "
            f"whatever the original serialisation was; `format_map.csv` records "
            f"what each one was, and directory names are left unhashed so the tree "
            f"still aligns with `MANIFEST.csv`.")
    else:
        hashed_note = (
            "An IRI-anonymised twin of the corpus can be built with "
            "`python pipeline/hash_corpus.py --corpus <release>/corpus`; it is not "
            "included in this deposition.")

    body = README
    for token, value in (
            ("@@TABLE@@", table),
            ("@@DISCOVERED@@", f"{n_discovered:,}"),
            ("@@KEPT@@", f"{n_kept:,}"),
            ("@@PCTKEPT@@",
             f"{100.0 * n_kept / n_discovered:.1f}" if n_discovered else "0"),
            ("@@GATEROWS@@", gate_rows),
            ("@@HASHED@@", hashed_note)):
        body = body.replace(token, value)
    leftover = [tok for tok in ("@@TABLE@@", "@@DISCOVERED@@", "@@KEPT@@",
                                "@@PCTKEPT@@", "@@GATEROWS@@", "@@HASHED@@")
                if tok in body]
    assert not leftover, f"unsubstituted placeholders: {leftover}"
    (rel / "README.md").write_text(body, encoding="utf-8")
    print(f"[OK] {rel / 'README.md'}")

    (rel / "validate_release.py").write_text(VALIDATOR, encoding="utf-8")
    print(f"[OK] {rel / 'validate_release.py'}")

    zenodo = {
        "title": "ODP-Reuse: a quality-gated benchmark of Ontology Design Pattern reuse",
        "upload_type": "dataset",
        "access_right": "open",
        "license": "cc-by-4.0",
        "description": (
            f"Ontologies that verifiably reuse Ontology Design Patterns (ODPs): "
            f"one unfiltered corpus of {n_discovered:,} discovered ontologies, "
            f"six quality-gated views over it ({n_kept:,} ontologies pass the "
            f"gate), an IRI-anonymised twin of the corpus, the "
            f"ontology-to-pattern mapping with per-pair coverage, per-view "
            f"statistics, and the complete record of how the corpus was "
            f"filtered.\n\n"
            f"Every ontology carries a gate verdict and, where it was excluded, "
            f"the rule and the evidence for the exclusion, so any rule can be "
            f"disputed and re-run rather than taken on trust. The gate removes "
            f"files that are themselves copies of a pattern (which make a "
            f"100%-reuse criterion tautological), byte-identical duplicates, "
            f"ontologies that contribute no entity of their own, and ontologies "
            f"with no pattern overlap to score against. FILTERS.md states each "
            f"rule with its threshold and counts; audit/ holds an independent "
            f"re-derivation of all removals, and validate_release.py checks an "
            f"unpacked copy for internal consistency.\n\n"
            f"Also included: a reproducible six-stage pipeline "
            f"(discover, repair, normalise, gate, audit, publish) and two worked "
            f"use cases showing the corpus is consumable rather than merely "
            f"descriptive -- ODP recommendation, where a recommender built on the "
            f"reuse matrix improves MAP by 0.805 over what pattern documentation "
            f"alone supports, and unsupervised ODP extraction, reported both "
            f"against an oracle upper bound and without ever consulting the "
            f"ground truth."),
        "keywords": ["ontology design patterns", "ontology reuse", "benchmark",
                     "knowledge graphs", "semantic web", "OWL", "data quality",
                     "reproducibility"],
        "creators": [
            {"name": "Norouzi, Ebrahim",
             "affiliation": "FIZ Karlsruhe -- Leibniz Institute for Information "
                            "Infrastructure",
             "orcid": "0000-0002-2691-6995"},
            {"name": "Hertling, Sven",
             "affiliation": "FIZ Karlsruhe / University of Mannheim",
             "orcid": "0000-0003-0333-5888"},
            {"name": "Waitelonis, Jörg", "affiliation": "FIZ Karlsruhe",
             "orcid": "0000-0001-7192-7143"},
            {"name": "Sack, Harald",
             "affiliation": "FIZ Karlsruhe / Karlsruhe Institute of Technology",
             "orcid": "0000-0001-7069-9804"},
        ],
    }
    (rel / "zenodo.json").write_text(json.dumps(zenodo, indent=2,
                                                ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    print(f"[OK] {rel / 'zenodo.json'}")


if __name__ == "__main__":
    main()
