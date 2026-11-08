# Get the data

<div class="odp-split" markdown>

<div class="odp-panel odp-panel--a" markdown>
### :material-database: Zenodo — the corpus
Every ontology and pattern, ~2.8 GB, versioned under one concept DOI that always
resolves to the newest release.

[:material-open-in-new: 10.5281/zenodo.17751740](https://doi.org/10.5281/zenodo.17751740){ .md-button .md-button--primary }
</div>

<div class="odp-panel odp-panel--b" markdown>
### :material-github: This repository — the tables
Views, manifest, audit and statistics: small, textual, diffable. Changes between
versions are reviewable as a pull request.

[:material-open-in-new: ISE-FIZKarlsruhe/ODP-Reuse](https://github.com/ISE-FIZKarlsruhe/ODP-Reuse){ .md-button }
</div>

</div>

## Why the split

The corpus contains single files above GitHub's 100 MB hard limit, and git would
retain every version of a 2.8 GB payload forever. Keeping the tables in git and
the corpus on Zenodo means a monthly update shows up as a readable diff instead
of an opaque binary churn.

## Use it in code

```python
import csv
from collections import defaultdict
from pathlib import Path
from rdflib import Graph

ds = Path("views/github-complete")

with (ds / "pairs.csv").open(encoding="utf-8") as fh:
    pairs = list(csv.DictReader(fh))

by_onto = defaultdict(list)
for r in pairs:
    by_onto[r["ontology_id"]].append(r)

oid = next(iter(by_onto))
for r in by_onto[oid]:
    pattern = Graph().parse(ds / r["pattern_file"])
    print(oid, "reuses", r["pattern"],
          f'{r["coverage_pct"]}% ({r["n_reused_entities"]}/{r["n_pattern_entities"]})')
```

Two conventions are not optional if you want comparable numbers: entity sets use
the **WIDE** definition, and annotation vocabularies are stripped from **both**
sides. Both live in `pipeline/odpqual/entities.py`, which every part of the
pipeline imports — so no two components can disagree about what an entity is.

## Verify your copy

```bash
python validate_release.py
```

Checks that every `pairs.csv` row resolves, that statistics match what the tables
actually contain, and that each `*-complete` view is a subset of its `*-reuse`
sibling.

## Cite

```bibtex
@dataset{odp_reuse_1_2_0,
  title  = {ODP-Reuse: a quality-gated benchmark of Ontology Design Pattern reuse},
  author = {Norouzi, Ebrahim and Hertling, Sven and
            Waitelonis, J{\"o}rg and Sack, Harald},
  year   = {2026},
  doi    = {10.5281/zenodo.17751740},
  version = {v1.2.0}
}
```
