# Changelog

Every release is a git tag, a GitHub Release and a Zenodo version under the same
concept DOI. The newest is shown first; all earlier versions remain resolvable.

_No changelog yet._

## Versioning

| Bump | Trigger | Comparability |
|---|---|---|
| **major** | the gate changes which ontologies are kept, or a view changes definition | earlier results **not** comparable |
| **minor** | new ontologies or pairs from a harvest; gate unchanged | earlier results valid on their subset |
| **patch** | repairs, metadata, documentation; membership unchanged | fully comparable |

## How updates happen

Two workflows test two different claims.

**The release reproduces.** `reproduce.yml` downloads the corpus archived on
Zenodo, checks every ontology file against the hash the gate recorded, re-runs
the gate and compares every row with the released `MANIFEST.csv`. It runs
monthly and on every change to the gate, so the claim that the archive, run
through the released gate, yields the released verdicts is tested rather than
asserted.

**The corpus grows; it never shrinks.** `quarterly-harvest.yml` re-runs
discovery, repair and the gate for the GitHub portion every quarter, searching
the release's pattern IRIs plus any the ODPA has published since. A re-harvest
queries live APIs and is a partial sample, not a census: code search returns at
most 1,000 files per query, and a file a harvest does not return is usually
still there. So a harvest only adds. `merge_harvest.py` keeps every released
row, file and verdict as it is, gates the ontologies the release does not have
against it (a new copy of a released file is a duplicate of the released one),
re-verifies their removals, and appends them to `MANIFEST.csv`, the views and
the audit. The workflow opens a pull request with the result. A person merges
it, adds the new files to the Zenodo corpus and tags a minor version; nothing
is published automatically.
