# Views

One directory per method, holding every group it proposed, flattened from the JSON Lines in `predictions/` so that the release is usable without a JSON parser.

`entities` is a `|`-separated list of entity IRIs. There is no score column: this corpus has no ground truth.

`review_shortlist.csv` is the ranked candidate set for expert review. Read `SIGNALS.md` before using its ordering.

| method | groups |
|---|---:|
| `all_in_one` | 103 |
| `asprino_cc` | 14,706 |
| `asprino_cc_full` | 610 |
| `connected_components` | 1,911 |
| `law_ap` | 1,463 |
| `law_cfp` | 172 |
| `namespace` | 584 |
| `owl2vec|hdbscan` | 2,111 |
| `owl2vec|kmeans` | 800 |
| `rdf2vec|hdbscan` | 2,057 |
| `rdf2vec|kmeans` | 640 |
| `singletons` | 54,344 |
