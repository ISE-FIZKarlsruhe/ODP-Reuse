# github-reuse

GitHub Code Search portion — **every ontology with at least one verified reused pattern entity**

Part of the **ODP-Reuse** release. This directory is self-contained: it holds the
ontologies, the ODP files they reuse, and the mapping between them.

## Contents

| Path | What it is |
|---|---|
| `ontologies/<id>/<id>.<ext>` | the ontology |
| `ontologies/<id>/patterns/*` | the ODP files this ontology reuses |
| `pairs.csv` | **the ontology-to-pattern mapping** — one row per (ontology, pattern) |
| `ontologies.csv` | per-ontology metadata (entity counts, reuse measures) |
| `statistics.csv`, `statistics.md` | the numbers reported in the paper for this dataset |
| `top_patterns.csv` | the most frequently reused patterns here |

## `pairs.csv` columns

| Column | Meaning |
|---|---|
| `ontology_id` | directory name under `ontologies/` |
| `pattern` | pattern stem (its file name without extension) |
| `pattern_file` | path to the pattern file, relative to this directory |
| `n_pattern_entities` | entities declared by the pattern |
| `n_reused_entities` | of those, how many occur in the ontology |
| `coverage_pct` | `100 * n_reused / n_pattern_entities` |
| `complete_reuse` | 1 if every pattern entity occurs in the ontology |

## Headline statistics

| Measure | Value |
|---|---:|
| Ontologies | 620 |
| Ontology–pattern pairs | 871 |
| Distinct patterns | 49 |
| Ontologies with a fully reused pattern | 42 |
| Fully reused pairs | 53 |
| Mean patterns per ontology | 1.4 |
| Mean pattern coverage (%) | 21.79 |
| Median entities per ontology | 69 |

## How this dataset was built

Every ontology here passed the quality gate in `pipeline/odpqual/gate.py`:

1. **G1** it parses as RDF;
2. **G2** it is *not itself a copy of a pattern* — its `owl:Ontology` IRI is not in
   a pattern-library namespace, it is not byte-identical to one of its patterns,
   and its entity set does not equal a pattern's;
3. **G3** it is not a byte-identical duplicate of another ontology (forks and
   mirrors collapse to one representative);
4. **G4** it contributes at least 10 entities of its own beyond its patterns;
5. **G5** at least one pattern has a non-empty entity overlap.


Rule G2 matters most. A 100%-exact-IRI criterion is *tautological* on a file that
is a verbatim copy of a pattern, and the previously published complete-reuse set
contained 14 such files out of 49. They are excluded here.

Entity sets use the WIDE definition (declared classes and properties, plus IRIs
occurring only in a class/property/individual position). Annotation vocabularies
(SKOS, DC, DCTERMS, CPANN, VANN, FOAF, …) are stripped from **both** the ontology
and the pattern before comparison, so the two sides are treated identically.

## Reproducing it

```bash
cd pipeline
python build_manifest.py --data-root <corpus> --out out
python build_release.py --manifest out/MANIFEST.csv --data-root <corpus> --out ../release
```

`../MANIFEST.csv` in this release records **every** ontology considered, including
the ones the gate dropped and why — so these counts can be audited rather than
taken on trust.
