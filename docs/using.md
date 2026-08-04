# Using the dataset

A practical guide: which view to load, what each column means, and the three
things that will otherwise catch you out.

## 1 · Pick a view

The release ships **six views** — three portions × two strictness levels. They
are not alternatives to be merged; each answers a different question.

| View | Ontologies | Pairs | Distinct patterns | Patterns/ontology |
|---|---:|---:|---:|---:|
| `github-reuse` | 442 | 669 | 45 | 1.51 |
| `github-complete` | 37 | 100 | 28 | 2.70 |
| `noOBO-reuse` | 51 | 90 | 45 | 1.76 |
| `noOBO-complete` | 6 | 16 | 11 | 2.67 |
| `OBO-reuse` | 8 | 758 | 420 | 94.75 |
| `OBO-complete` | 8 | 758 | 420 | 94.75 |

A `-reuse` view contains every ontology that reuses **at least one entity** of at
least one pattern. The matching `-complete` view keeps only those that reuse a
pattern **in full**, and is the stricter, smaller ground truth.

<div class="odp-note" markdown>
**Which one do I want?**

- Training a recommender, or measuring *how much* is reused → a `-reuse` view.
- Evaluating pattern *detection*, where a partial hit is not a hit → `-complete`.
- Reporting a single headline number over the whole corpus → don't. See §4.
</div>

## 2 · The files and their columns

Each view directory holds `pairs.csv`, `ontologies.csv`, `statistics.csv` and
`top_patterns.csv`. The two that matter:

=== "pairs.csv — one row per (ontology, pattern)"

    | Column | Meaning |
    |---|---|
    | `ontology_id` | stable id; also the directory name under `corpus/` |
    | `pattern` | pattern stem, e.g. `agentrole` |
    | `n_pattern_entities` | entities the **pattern** declares |
    | `n_reused_entities` | of those, how many appear in the ontology |
    | `coverage_pct` | `100 × n_reused / n_pattern_entities` — see §4 |
    | `complete_reuse` | `1` when coverage is 100% |

=== "ontologies.csv — one row per ontology"

    | Column | Meaning |
    |---|---|
    | `n_entities_wide` | all entities, including those only referenced |
    | `n_own_entities` | entities contributed **beyond** its patterns |
    | `n_gt_patterns` | patterns with a non-empty intersection |
    | `max_reuse_pct` | best coverage achieved over its patterns |

`MANIFEST.csv` at the release root is the **unfiltered** record: all 1,973
discovered ontologies, each with `stage_reached` and `drop_reason`. The gate is
recorded as metadata, never applied destructively — so you can disagree with any
rule and re-derive your own corpus from the manifest alone.

## 3 · Loading it

```python
import pandas as pd

pairs = pd.read_csv("release-iri/views/github-reuse/pairs.csv")

# ontology x pattern reuse matrix, the usual starting point
M = pairs.pivot_table(index="ontology_id", columns="pattern",
                      values="n_reused_entities", fill_value=0)

# ... or restrict to full reuse only
strict = pairs[pairs.complete_reuse == 1]
```

To re-derive the corpus under **your own** gate rather than ours:

```python
man = pd.read_csv("release-iri/MANIFEST.csv")
mine = man[(man.parsed == 1) & (man.n_own_entities >= 25)]   # your threshold
```

## 4 · Three things that will catch you out

!!! warning "`coverage_pct` is a share of the *pattern*, not of the ontology"
    It answers "how much of this pattern is present?", so a large ontology
    reusing one small pattern in full scores 100%. It is **not** a measure of how
    pattern-based the ontology is.

!!! danger "Never pool one average across the three portions"
    The DOS-DP portion is 8 ontologies against 442 from GitHub and
    51 from Paper &amp; Repository, but its ontologies carry orders of
    magnitude more entities and its patterns are generated from YAML templates
    rather than chosen by a modeller. A micro-average over the pooled corpus is
    dominated by it and reverses conclusions that hold per portion. Report the
    three separately, and say which averaging you used.

!!! note "The corpus is unfiltered by design"
    1,973 ontologies ship; 501 pass the gate. If you load `corpus/`
    directly instead of a view, you are working with everything — including
    unparseable files, verbatim pattern copies and duplicates. Join against
    `MANIFEST.csv` on `ontology_id` and filter on `stage_reached`.

## 5 · Plain or hashed?

Each dataset has an IRI-anonymised twin (`corpus-hashed`, and `-hashed` siblings
of the views). Every IRI is replaced by a deterministic hash, so identical
entities stay aligned across files, but pattern namespaces are unrecognisable.

- **Evaluating a detection method** → use the hashed form. On the plain form a
  method can score well by recognising `ontologydesignpatterns.org` prefixes
  instead of learning structure.
- **Inspecting, curating or authoring** → use the plain form.

Hashed corpus files are N-Triples regardless of the original serialisation, and
`format_map.csv` records what each one was. Directory names are *not* hashed, so
the tree still aligns with the manifest.

## 6 · Citing it

Cite the Zenodo DOI for the data and the paper for the method — see
[Get the data](download.md).
