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

A scheduled workflow re-runs discovery, repair, de-duplication and the gate every
quarter, then opens an issue listing ontology–pattern pairs that did not exist in
the published release, and a pull request with the updated tables. If the pipeline
cannot be re-run from a clean checkout, the workflow fails publicly — the
reproducibility claim is tested rather than asserted.
