# ODP-Reuse — a quality-gated benchmark of Ontology Design Pattern reuse

Which ontologies **actually** reuse Ontology Design Patterns, and how much of each
pattern do they reuse? This repository harvests candidate ontologies from code
repositories, publications and ontology portals, verifies reuse entity by entity,
and publishes the result as an **unfiltered corpus plus six quality-gated views**.

Every ontology carries a gate verdict and, where it was excluded, the rule and the
evidence for the exclusion — so any rule can be disputed and re-run rather than
taken on trust.

| | |
|---|---|
| Ontologies harvested | **1,973** |
| Pass the quality gate | **501** (25.4%) — GitHub 442 · Paper & Repository 51 · DOS-DP 8 |
| Verified reuse pairs | **1,517** over **488** distinct patterns |
| Reuse a pattern *in full* | **51** ontologies |

📖 **[Full documentation](https://ise-fizkarlsruhe.github.io/ODP-Reuse/)** ·
💾 **[Data on Zenodo](https://doi.org/10.5281/zenodo.17751740)** ·
🚀 **[How to use the dataset](https://ise-fizkarlsruhe.github.io/ODP-Reuse/using/)**

## Get the data

The corpus itself (several GiB of OWL) lives on Zenodo. The **tables** — the
ontology-to-pattern mapping, per-view statistics, the manifest and the audit — are
in this repository under `release-iri/` and are what most users need:

```python
import pandas as pd
pairs = pd.read_csv("release-iri/views/github-reuse/pairs.csv")
M = pairs.pivot_table(index="ontology_id", columns="pattern",
                      values="n_reused_entities", fill_value=0)
```

`release-iri/MANIFEST.csv` is the unfiltered record of all 1,973 ontologies with
`stage_reached` and `drop_reason`, so you can re-derive a corpus under your own
gate instead of ours. See [Using the dataset](https://ise-fizkarlsruhe.github.io/ODP-Reuse/using/)
for the column reference and the three traps worth knowing about — chiefly that
`coverage_pct` is a share of the *pattern*, and that the three portions must never
be pooled into one average.

## Layout

| Path | What it is |
|---|---|
| `release-iri/` | the release: manifest, six views, gate summary, audit evidence |
| `pipeline/` | the six-stage pipeline — discover · repair · normalise · gate · audit · publish |
| `use-cases/odp-extraction/` | UC1 — the corpus as **ground truth** for unsupervised ODP extraction |
| `use-cases/odp-recommendation/` | UC2 — the corpus as **training signal** for ODP recommendation |
| `MultiSource/`, `scholar/` | the harvesters: ontology discovery, and the ODP literature survey |
| `docs/` | the documentation site (MkDocs), generated from the release |

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The gate and the release are rebuilt from the corpus with:

```bash
python pipeline/build_manifest.py  --corpus <corpus-root> --out pipeline/out
python pipeline/audit_filters.py   --manifest pipeline/out/MANIFEST.csv
python pipeline/build_release.py   --manifest pipeline/out/MANIFEST.csv --out release-iri
python pipeline/gen_site.py                       # regenerate docs/ from the release
```

Two checks guard against drift, and both should pass before any commit:

```bash
python paper-iri/verify_numbers.py    # every headline number vs the released CSVs
python pipeline/check_repo_claims.py  # every path the manuscript cites is present
```

`verify_numbers.py` reports how many assertions it ran — 108 in a fresh clone,
140 in a working tree that also holds the unpublished companion track. It is the
reason four errors in the tables were caught rather than published.

Harvesting is **not** re-runnable to the same result: it queries live APIs
(GitHub, Google, BioPortal, MatPortal, LOV), so the reproducible artefact is the
archived Zenodo deposition, not a re-run of the harvester. Credentials for a fresh
harvest go in `.env` — see `.env.example`.

## Use cases

Two demonstrations that the corpus is *consumable*, not merely descriptive:

- **UC1 · extraction** — recover ODP instances by embedding and clustering, with
  no knowledge of which patterns an ontology reuses. On 442 gate-passing GitHub
  ontologies the best unsupervised configuration reaches F₁ **0.321**, where an
  oracle that tunes against the answer reaches **0.453**. Both are reported,
  because only one of them is achievable.
- **UC2 · recommendation** — recommend the ODPs a new ontology should reuse. A
  recommender built on the reuse matrix beats the best that pattern documentation
  alone supports by **+0.805 MAP** (95% CI [+0.729, +0.874], 83/3/3 wins,
  sign test *p* < 0.001, *n* = 89 paired ontologies).

Read `use-cases/*/CAVEATS.md` **before citing any number** — each records which
claims survive a paired significance test and which do not.

## Citation

Cite the dataset via its Zenodo concept DOI, which always resolves to the latest
version: [10.5281/zenodo.17751740](https://doi.org/10.5281/zenodo.17751740).
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). The accompanying
paper is under review; its citation will be added here once it is accepted.

## License

Code is [MIT](LICENSE). The **data** — the corpus, the views and the derived
tables — is **CC BY 4.0**, as recorded in `CITATION.cff` and on Zenodo. The
third-party pattern libraries the pipeline reads are not redistributed here; each
keeps its own licence.
