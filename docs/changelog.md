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

**The corpus is maintained.** `quarterly-harvest.yml` re-runs discovery, repair
and the gate for the GitHub portion every quarter and opens an issue listing
ontology–pattern pairs that did not exist in the published release; hits for
patterns published after the release are reported separately. A re-harvest
queries live APIs and is not expected to reproduce the corpus: it tests that the
harvester still runs, and shows what is new. A new release is cut deliberately
from that issue, never automatically.
