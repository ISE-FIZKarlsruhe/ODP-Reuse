# ODP-Reuse: a quality-gated benchmark of ontology design pattern reuse

Ontologies that verifiably reuse Ontology Design Patterns (ODPs): one unfiltered
corpus, six quality-gated views over it, an IRI-anonymised twin of the corpus, the
ontology-to-pattern mapping, per-view statistics, and the complete record of how
the corpus was filtered.

Every ontology here survived a documented quality gate. Most candidates did not:
of **3,705** ontologies discovered, **679** (18.3%) are usable.
That attrition is itself a finding, and `MANIFEST.csv` records the reason for
every exclusion.

## The six views

| Dataset | Ontologies | Pairs | Distinct patterns | Fully reused pairs |
|---|---:|---:|---:|---:|
| `github-reuse` | 620 | 871 | 49 | 53 |
| `github-complete` | 42 | 108 | 31 | 53 |
| `noOBO-reuse` | 51 | 90 | 45 | 6 |
| `noOBO-complete` | 6 | 16 | 11 | 6 |
| `OBO-reuse` | 8 | 758 | 420 | 620 |
| `OBO-complete` | 8 | 758 | 420 | 620 |

`*-reuse` — every gated ontology with at least one verified reused pattern entity.
`*-complete` — the subset where at least one pattern is reused *in full*.

`corpus-hashed/` is the IRI-anonymised twin of `corpus/`: 9,884 files, same content and same reuse relations, every IRI replaced by a deterministic hash. Use **hashed to evaluate** a detection method — it removes the namespace-recognition shortcut — and **plain to inspect or curate**. Its files are N-Triples whatever the original serialisation was; `format_map.csv` records what each one was, and directory names are left unhashed so the tree still aligns with `MANIFEST.csv`.

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
  ontology (OWL2Vec\*, RDF2Vec), clusters, and scores clusters against the ground
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
| **G1** | parses as RDF | 425 | an unparseable file cannot be analysed |
| **G2** | **is not itself a copy of a pattern** — its `owl:Ontology` IRI is not in a pattern-library namespace, it is not byte-identical to one of its patterns, and its entity set does not equal a pattern's | 51 | a 100%-reuse criterion is *tautological* on a file that **is** the pattern |
| **G3** | is not a byte-identical duplicate of another ontology | 342 | forks and mirrors inflate counts and leak across train/test splits |
| **G4** | contributes at least **1** entity of its own, beyond its patterns | 1,652 | an ontology that adds nothing to its patterns is an aggregation of them, not a reusing ontology; every ontology dropped here contributes exactly zero |
| **G5** | has ≥1 pattern with a non-empty entity overlap | 556 | no overlap means no reuse signal and nothing to score against |
| **G6** | has ≥1 pattern reused *in full* (the `*-complete` views only) | 0 | applied *after* G1–G5, so it cannot be satisfied by a pattern copy |

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
python run_rq_stats.py    --manifest out/MANIFEST.csv --data-root <corpus> \
                          --out out/rq_iri_gated
python build_release.py   --manifest out/MANIFEST.csv --data-root <corpus> \
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
