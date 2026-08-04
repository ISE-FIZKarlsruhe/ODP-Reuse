# Datasets

Three corpus portions × two reuse levels. A **view** is a filter over the
published corpus, not a copy of it — so nothing is duplicated and you can always
inspect what a view left out.

| View | Ontologies | Pairs | Patterns | Complete pairs | Mean patterns/onto | Median entities |
|---|---:|---:|---:|---:|---:|---:|
| [`github-reuse`](download.md) | 442 | 669 | 45 | 47 | 1.51 | 61 |
| [`github-complete`](download.md) | 37 | 100 | 28 | 47 | 2.70 | 122 |
| [`noOBO-reuse`](download.md) | 51 | 90 | 45 | 6 | 1.76 | 43 |
| [`noOBO-complete`](download.md) | 6 | 16 | 11 | 6 | 2.67 | 11 |
| [`OBO-reuse`](download.md) | 8 | 758 | 420 | 620 | 94.75 | 32,095 |
| [`OBO-complete`](download.md) | 8 | 758 | 420 | 620 | 94.75 | 32,095 |

<div id="odp-views" class="odp-viz" data-src="data/views.json"></div>

## Reading a view

Each view directory contains `pairs.csv` — the ontology-to-pattern mapping, one
row per pair — plus per-ontology metadata and statistics.

| Column | Meaning |
|---|---|
| `ontology_id` | directory name under `corpus/` |
| `pattern` | pattern stem |
| `n_pattern_entities` | entities declared by the pattern |
| `n_reused_entities` | of those, how many occur in the ontology |
| `coverage_pct` | `100 × reused / pattern entities` |
| `complete_reuse` | 1 if every pattern entity occurs in the ontology |

!!! danger "DOS-DP is not comparable to the others"
    Its ontologies carry a median of tens of thousands of entities against a
    median of ~70 elsewhere, and its patterns are generated from YAML templates
    rather than chosen. Report it separately; never pool a single averaged figure
    across all three portions.
