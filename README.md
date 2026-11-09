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

### The release reproduces from the archive

What reproduces exactly is the release: the corpus archived on Zenodo, run
through the released gate, yields `release-iri/MANIFEST.csv` row for row.
[`reproduce.yml`](.github/workflows/reproduce.yml) checks this monthly and on
every change to the gate. A value corrected after its release (CHANGELOG.md
names each correction) is checked as corrected: the job compares the published
version's rows and lists those that main has since corrected to exactly what
the re-run gives.

The gate runs in the order the release was built. Each version after v1.2.0
added ontologies and gated them against those already published, so a new copy
of a published file is a duplicate of the published row even when its id sorts
first. `--additions` replays that: first the ontologies no
`release-iri/additions/v*.csv` lists, then each version's additions in turn.
Without it, 41 rows of v1.3.0 would differ: they are 15 groups of identical
files, and run all at once the gate would keep the new copy's id. By hand:

```bash
pip install -r pipeline/requirements-reproduce.txt     # the pinned RDF parser
record=$(python -c "import json; print(json.load(open('release-iri/archive.json'))['zenodo_record'])")
curl -L -o corpus.zip "https://zenodo.org/api/records/$record/files/corpus.zip/content"
unzip -q corpus.zip                                     # -> corpus/{github,noOBO,OBO}/

# 1. every archived ontology file is byte-identical to the one the gate hashed
python pipeline/verify_reproduction.py hashes --manifest release-iri/MANIFEST.csv --corpus corpus

# 2. re-running the gate gives the released verdicts
mkdir regate-root
ln -s "$PWD/corpus/github" regate-root/github_dataset   # the gate's name for it
ln -s "$PWD/corpus/noOBO"  regate-root/noOBO
ln -s "$PWD/corpus/OBO"    regate-root/OBO
python pipeline/build_manifest.py --data-root regate-root --out regate --max-mb 100
python pipeline/verify_reproduction.py compare --release release-iri/MANIFEST.csv \
    --regated regate/MANIFEST.csv --additions release-iri/additions
```

`--max-mb 100` skips re-parsing the five largest ontologies (`noOBO/cco` alone is
1.1 GB and 15.7 M triples, far more than a 16 GB machine can hold in rdflib);
step 1 still checks their files. `--max-mb 0` re-parses everything. On Windows,
unpack to a short path such as `C:\c`: without long-path support, ontology paths
under a deep folder exceed 260 characters and silently fail to open.

v1.2.0, the last version gated all at once, is rebuilt from a gated manifest
with the commands below; later versions add to it with `merge_harvest.py`
(see [Re-harvesting](#re-harvesting-finds-what-is-new-it-does-not-reproduce)):

```bash
python pipeline/audit_filters.py --manifest regate/MANIFEST.csv --data-root regate-root --out regate/audit
python pipeline/build_release.py --manifest regate/MANIFEST.csv --data-root regate-root --out release-iri
python pipeline/gen_site.py                             # regenerate docs/ from the release
```

The harvester and the reproduction checks have tests:

```bash
python -m pytest MultiSource/tests pipeline/tests
```

### Re-harvesting finds what is new; it does not reproduce

Harvesting is **not** re-runnable to the same result: it queries live APIs
(GitHub, BioPortal, MatPortal, LOV), so the reproducible artefact is the
archived Zenodo deposition, not a re-run of the harvester.
[`harvest.yml`](.github/workflows/harvest.yml) re-harvests
the GitHub portion every month, searching the release's 144 pattern IRIs plus
any the ODPA repository has published since, labelled separately. Files in the
pattern libraries themselves (`odpa/patterns-repository`, and this repository's
own pattern copies under `docs/patterns/`) are not kept: they publish the
patterns rather than reuse them, and the release holds none of them.

A harvest only **adds** to the release. Code search returns at most 1,000 files
per query and ranks them differently on every run, so a harvest that does not
return a released ontology says nothing about it: after the first full harvest,
88 of 100 sampled "missing" files were still at the same path.
[`pipeline/merge_harvest.py`](pipeline/merge_harvest.py) keeps every released
row, file and verdict, and gates only the ontologies the release does not have
(against the release, so a new copy of a released file is a duplicate of the
released row):

```bash
cd pipeline
python merge_harvest.py --release ../release-iri \
    --harvest-manifest ../harvest/out/MANIFEST.csv --harvest-root ../harvest \
    --additions ../harvest/corpus-additions --report ../harvest/merge
python write_release_readme.py --release ../release-iri --keep-hashed-note
```

When a month's harvest adds something,
[`publish-release.yml`](.github/workflows/publish-release.yml) publishes it as
the next minor version with no manual step: a Zenodo version under the concept
DOI, a git tag, a GitHub Release, and an entry in [`CHANGELOG.md`](CHANGELOG.md)
and `release-iri/history.csv`, from which the
[changelog page](https://ise-fizkarlsruhe.github.io/ODP-Reuse/changelog/) draws
the dataset's history. It publishes nothing unless every check passes: the
update only adds, the gate is the released gate, the new version builds on the
archived one, and every file in the new archive matches the hash in
`MANIFEST.csv`. A dry run checks and assembles everything and publishes
nothing:

```bash
gh workflow run publish-release.yml -f run_id=<harvest run id> -f dry_run=true
```

Google's Custom Search JSON API, used for the original harvest, shuts down on
2027-01-01. Credentials for a fresh harvest go in `.env` — see `.env.example`.

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
