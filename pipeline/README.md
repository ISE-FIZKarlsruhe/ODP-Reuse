# Reproducible pipeline: quality gate, RQ1–RQ5

This directory rebuilds every ontology count and every RQ table in the paper from
the raw corpus, with no dependency on the PatternMiner repository. It exists
because the published counts did not reconcile: the paper reported 1,658 GitHub
ontologies while PatternMiner evaluated a differently-sized corpus, DOS-DP
results existed in the paper but not in PatternMiner's aggregates, and the
complete-reuse "gold standard" contained pattern copies matched against
themselves.

**One rule governs everything here: no count appears in any manuscript that is
not a documented filter over `MANIFEST.csv`.**

---

## Layout

```
pipeline/
├── odpqual/
│   ├── entities.py          entity sets (NARROW/WIDE) + annotation stripping
│   └── gate.py              the quality gate, rules G1–G6
├── odpstats/
│   └── rq_stats.py          RQ1–RQ4 measures
├── build_manifest.py        → MANIFEST.csv, gate_summary.{csv,md}
├── run_rq_stats.py          → RQ1–RQ5 CSVs + LaTeX tables
└── run_uc_recommendation.py → use-case tables for ODP recommendation

The two use cases live outside this directory, each self-contained, and import
the gate from here so all three agree on what an entity is:

    ../use-cases/odp-recommendation/   the dataset as training signal
    ../use-cases/odp-extraction/       the dataset as ground truth
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`gensim` imports `scipy.linalg.triu`, removed in scipy ≥ 1.12. The embedders call
`_common.scipy_triu_patch()` before importing `gensim`, so no pin is needed —
but if you import `gensim` yourself first, apply the shim.

---

## Data layout

The corpus is three sibling directories under one root, each holding one
directory per ontology:

```
<data-root>/
├── github_dataset/<ontology>/<ontology>.owl + patterns/*.owl
├── noOBO/<ontology>/...        Paper & Repository portion
└── OBO/<ontology>/...          DOS-DP portion
```

Dataset keys used everywhere: `github`, `noOBO`, `OBO`.

---

## Step 1 — the quality gate

```bash
python build_manifest.py \
    --data-root /path/to/MultiSource \
    --out out \
    --datasets github noOBO OBO
```

Writes `out/MANIFEST.csv` (one row per ontology, every gate column),
`out/gate_summary.{csv,md}`, and `out/cache/<dataset>.jsonl`. Pass `--resume` to
continue an interrupted run from the cache.

### The rules, and why each exists

| Rule | Test | Justification |
|---|---|---|
| **G1** parseable | the ontology file parses as RDF | an unparseable file cannot be analysed; ~14% of the GitHub portion fails |
| **G2** not a pattern copy | its `owl:Ontology` IRI is not in a pattern-library namespace; it is not byte-identical to one of its patterns; its entity set does not equal a pattern's | **the decisive rule.** A 100%-exact-IRI criterion is tautological on a file that *is* the pattern. 14 of the 49 previously published complete-reuse ontologies are verbatim copies of `bag.owl`, `list.owl`, `agentrole.owl`, … |
| **G3** not a duplicate | among identical content hashes, keep the lexicographically first | 11% of the GitHub portion is byte-identical fork/mirror copies; duplicates inflate counts and leak across splits |
| **G4** substantive | contributes ≥ `--min-own-entities` (default 10) entities not supplied by its patterns | a third of the GitHub portion is config/metadata stubs that cannot exhibit reuse |
| **G5** has ground truth | ≥ 1 pattern with non-empty entity intersection | required for evaluation |
| **G6** complete reuse | ≥ 1 pattern fully contained, evaluated **after** G1–G5 | the original criterion, now unsatisfiable by a file being a copy of the pattern |

Each ontology is attributed to the **first** rule it fails, so the
`gate_summary` columns partition `discovered` exactly.

### Two deliberate choices

**No size limit on the dataset.** A 1 GB ontology is a legitimate corpus member,
so `--max-mb` defaults to `0` (unlimited). Size constrains only the *embedding*
stage, where it is recorded as a skip reason rather than a dataset exclusion.

**Annotations are stripped before analysis.** PatternMiner filtered
annotation vocabularies (SKOS, DC, DCTERMS, CPANN) on the *pattern* side but not
on the *host ontology* side, so `dcterms`/`skos` edges became random-walk edges
and KGE triples for hosts but not for patterns — biasing every host-vs-pattern
comparison. `odpqual.entities.strip_annotations` removes them from both sides.
Pass `--keep-annotations` to reproduce the old asymmetric behaviour.

### Entity definitions

`collect_entities` is byte-compatible with PatternMiner's
`rq_0_cache/build_cache.py::_collect_entities`, so counts reconcile across the
two papers. `WIDE` (the default, matching PatternMiner's clean run which set
`ODP_WIDE_ENTITIES=1`) additionally collects IRIs occurring only in a
class/property/individual position. Validation: on `noOBO` this gate keeps **38**
ontologies, exactly matching `gt_quality_noOBO.csv`'s `ok = 38`.

---

## Step 2 — RQ1–RQ4

```bash
python run_rq_stats.py \
    --manifest out/MANIFEST.csv \
    --data-root /path/to/MultiSource \
    --out out/rq
```

| Output | Content |
|---|---|
| `pairs.csv` | one row per (ontology, pattern): coverage %, role counts, DL labels, constructs |
| `per_ontology.csv` | per-ontology entity counts and entities added beyond the patterns |
| `rq1_extent.csv` | reuse and complete-reuse counts, threshold distribution, mean patterns/ontology |
| `rq1_top_patterns.csv` | most reused patterns per dataset |
| `rq2_reuse_forms.csv` | eight structural reuse roles as shares of role occurrences |
| `rq3_complexity.csv` | pattern vs ontology DL expressivity; constructs only on the ontology side |
| `rq4_expansion.csv` | classes / object / datatype properties added beyond the patterns |
| `tables/*.tex` | the LaTeX tables, generated from the CSVs |

DL labels are construct-based and approximate — they order corpora, they do not
certify them. A reasoner is required for a certified label.

---

## Step 3 — RQ5, automatic ODP extraction

Only the two embedders named in the paper are run: **OWL2Vec\*** and **RDF2Vec**,
each with **K-Means** and **HDBSCAN**. The 45-method sweep lives in PatternMiner
and is out of scope.

```bash
# the small portions
python run_rq5_extraction.py --manifest out/MANIFEST.csv \
    --data-root /path/to/MultiSource --out out/rq5 \
    --datasets github noOBO --subset reuse

# DOS-DP, uncapped (very large ontologies; expect hours)
python run_rq5_extraction.py --manifest out/MANIFEST.csv \
    --data-root /path/to/MultiSource --out out/rq5_obo \
    --datasets OBO --subset complete --max-entities 0 --max-mb 0
```

`--subset complete` restricts to the complete-reuse subset (the paper's choice);
`--subset reuse` uses every gated ontology with any verified reuse.

### Two numbers per configuration, and why

PatternMiner selects clustering hyper-parameters per ontology by argmax of
entity-F1 **against the ground truth** (`rq_0_cache/rich_eval.py:312/392`). That
is an oracle: it consults the labels it is scored against, so the resulting F1 is
an upper bound, not an achievable score.

Every configuration is therefore reported twice:

- **`oracle_*`** — best F1 over the grid, chosen using the ground truth.
  Comparable to the previously published numbers. Label it an upper bound.
- **`unsup_*`** — chosen by silhouette, never touching the ground truth. The
  honest, achievable number.
- **`oracle_gap_f1`** — the difference, which quantifies how much of the
  published performance came from oracle model selection.

### Nothing is skipped silently

Any ontology excluded from RQ5 is written to `rq5_skipped.csv` with a reason:
`file_too_large`, `parse_error`, `no_entities`, `too_many_entities`,
`no_ground_truth`, `embed_out_of_memory`, `embed_error`. Silent drops are how the
original count inconsistencies arose; a skip is data, not noise.

`embed_out_of_memory` is expected on the largest ontologies — OWL2Vec\* and
RDF2Vec build `n_walks` walks per entity, so memory grows with entity count. The
ontology stays in the dataset; only this machine's embedding of it fails.

---

## Reconciling with PatternMiner

The gate was written independently of PatternMiner's `gt_recovery.py`, and the
failure modes align:

| gate rule | count | `gt_quality_github.csv` | count |
|---|---:|---|---:|
| G1 parse | 268 | `parse_error` | 234 |
| G4 substantive | 716 | `too_few_entities` | 626 |
| G5 has GT | 439 | `no_match` | 411 |
| kept | 242 | `ok` | 202 |
| G3 duplicate | 209 | *(no dedup exists upstream)* | — |

Residual differences are accounted for: PatternMiner audits 1,784 of 1,887
ontologies; it has no deduplication; and its stub criterion differs from G4's
≥10-own-entities threshold.

**Before citing any PatternMiner number**, check
`PatternMiner/aggregates/PROVENANCE.md`. Note that
`per_ontology_f1_{combined,github,noOBO}.csv` currently hold +194/+164/+30 rows
more than PROVENANCE documents, so that citation gate is stale for exactly the
files it clears.

---

## Determinism

All random seeds are fixed (embedders seed 42, K-Means `random_state=42`,
bootstrap seed 0). Re-running on the same corpus reproduces the same numbers.
The corpus itself is a snapshot: the discovery pipeline queries live APIs, so
re-harvesting will not reproduce it — use the archived Zenodo release.
