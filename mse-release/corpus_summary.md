# Corpus summary

This release has **no quality gate and no ground truth**, so there is no attrition table of the kind `release-iri/gate_summary.md` carries. What determines the population here is whether a file parses, whether it declares any entity, and what each run's cost bound admits. All three are recorded below and per-ontology in `MANIFEST.csv` and `skipped.csv`.

## Composition

| measure | files |
|---|---:|
| in the corpus | 116 |
| parse as RDF | 114 |
| do not parse | 2 |
| refreshed from upstream for this release | 29 |

## Upstream currency audit

Every parseable file was checked against its upstream before evaluation, and each proposed update was independently re-checked before it was applied. `provenance/upstream_check.json` holds the verdict for every file, including those where nothing newer was found.

| verdict | files |
|---|---:|
| could not be established | 1 |
| newer release found | 43 |
| no reachable upstream | 7 |
| already current | 63 |

Of the artefacts with a newer release, 29 were applied; the rest were rejected by the re-check or by the entity-count guard, and `provenance/corpus_updates.csv` records each decision with its reason.

## Exclusions

Every ontology a run could not use, with the reason. An exclusion is data, not noise: a silent drop is how a population quietly stops being the one a reader thinks it is.

| reason | records |
|---|---:|
| `no_ground_truth_predict_only` — no answer key, so predicted but not scored (expected here) | 204 |
| `no_entities` — parses but declares no entity (an import-only shell) | 6 |
| `parse_error` — the file is not parseable RDF | 4 |
| `too_many_entities` — above the embedding cost bound | 2 |
| `asprino_too_many_entities` — above the conceptual-components cost bound | 1 |

