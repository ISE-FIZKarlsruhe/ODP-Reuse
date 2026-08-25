# Use case: automatic ODP extraction

**Question this use case answers:** *can the ODP-Reuse dataset serve as ground
truth for recovering design patterns from ontologies automatically?*

Given an ontology and no knowledge of which patterns it reuses, an extraction
method must propose groups of entities that correspond to reused ODP instances.
The dataset supplies the answer key: for every ontology it records which pattern
entities are actually present. That makes extraction measurable rather than
arguable.

This is the second of two reusability demonstrations. The other is
[`../odp-recommendation/`](../odp-recommendation/), which uses the dataset as
*training signal* rather than as ground truth.

> **Read [`CAVEATS.md`](CAVEATS.md) before citing any number from this
> directory.** It is the citation gate.

---

## What is here

```
use-cases/odp-extraction/
├── README.md                     this file
├── CAVEATS.md                    citation gate — what may and may not be cited
├── requirements.txt              Python dependencies
│
├── extractbench/                 the benchmark
│   ├── owl2vec.py                ┐ vendored verbatim from PatternMiner so both
│   ├── rdf2vec.py                │ papers use byte-identical embedders
│   ├── _common.py                ┘
│   ├── asprino.py                Asprino et al. (K-CAP 2021) step 1, adapter
│   ├── lawrynowicz/              Ławrynowicz et al. (SWJ 2018), reimplemented
│   │   ├── axiom_trees.py        §4.2  ontology → forest of labelled trees
│   │   ├── treemine.py           §4.3–4.5  frequent induced unordered subtrees
│   │   ├── cfp.py                §4.6  maximal frequent itemsets → class frames
│   │   └── method.py             reading a mined pattern back onto an ontology
│   ├── structural.py             namespace / connected-components / degenerate
│   └── evaluate.py               clustering + Hungarian evaluation
│
├── third_party/
│   └── conceptual_components/    Asprino et al.'s own step-1 code, vendored
│       └── PROVENANCE.md         upstream commit, blob hashes, every patch
│
├── results/<dataset>/            per-ontology results and aggregates
├── stats/uc1/                    the combined UC1 table (CSV + LaTeX)
└── tools/
    ├── run_extraction_eval.py    the embedding runner
    ├── run_method_eval.py        the published-methods + reference-points runner
    ├── make_uc1_tables.py        joins both runners into one paired table
    └── verify_asprino_port.py    checks the vendored port against its own
                                  authors' published output
```

## The two published methods

The resource exists because {Ł}awrynowicz et al. and Asprino et al. both report
that no reference dataset of confirmed pattern reuse was available to score
against. Both are run here.

| | source | what is run | what is **not** run |
|---|---|---|---|
| **Asprino et al.** (K-CAP 2021) | the authors' own code, vendored | step 1: intensional graph → greedy modularity communities, recursively refined below average density | steps 2–7, which cluster communities *across* ontologies into a corpus-level catalogue. That is a different output from the one this benchmark scores, and needs a self-built UKB binary and a 1.6 GB Framester jar. |
| **{Ł}awrynowicz et al.** (SWJ 2018) | reimplemented — the published FF-SLEUTH URL (`bitbucket.org/leolod/jsleuth`) is HTTP 404 | axiom-tree encoding, frequent unordered **induced** subtree mining over a family of forests (σ=4 ontologies), maximality + axiom-operator filters, propositionalisation, maximal frequent itemsets (σ_CF=4) | decoding to Manchester syntax with variable completion — cosmetic here, since it does not change which entities a pattern matches |

Never write "Asprino et al. scores X"; write "Asprino et al. (step 1)". The port's
fidelity is checked, not asserted: `tools/verify_asprino_port.py` re-runs it on
the authors' Conference corpus and reproduces 423 communities against their
published 419, 6 of 16 ontologies exactly.

Turning either method's output into a set of entities is a reading this
benchmark fixes and neither paper does; `extractbench/lawrynowicz/method.py`
states it, and both readings are computed and released.

The quality gate and the dataset registry live in [`../../pipeline/`](../../pipeline/),
not here, because the two use cases and the RQ statistics must agree on them. The
runner adds `pipeline/` to `sys.path` automatically.

---

## The task

**Input:** one ontology, as an RDF graph. Nothing about its patterns.

**Output:** a set of predicted clusters, each a set of entity IRIs.

**Ground truth:** for each pattern the ontology reuses, the set of that pattern's
entities present in the ontology — exactly what `pairs.csv` records in the release.

**Pipeline:** random walks over the ontology graph → Word2Vec embeddings
(OWL2Vec\* or RDF2Vec) → clustering (K-Means or HDBSCAN) → each cluster is a
candidate pattern instance.

**Scoring.** Predicted clusters are matched to ground-truth patterns by Hungarian
assignment on the cost matrix `1 − F1`, then entity-level precision, recall and
F1 are macro-averaged over matched pairs per ontology, and over ontologies:

```
precision = |p ∩ g| / |p|      recall = |p ∩ g| / |g|      F1 = 2PR/(P+R)
```

Only the two embedders named in the paper are run: **OWL2Vec\*** and **RDF2Vec**,
each with **K-Means** and **HDBSCAN**. PatternMiner's 45-method sweep is a
separate study and out of scope here.

---

## Two numbers per configuration, and why it matters

Every configuration is reported **twice**:

| Regime | Hyper-parameters chosen by | Interpretation |
|---|---|---|
| `oracle_*` | argmax entity-F1 **against the ground truth** | an upper bound, not an achievable score |
| `unsup_*` | silhouette, never touching the ground truth | the honest, achievable number |
| `oracle_gap_f1` | the difference | how much of the score came from oracle model selection |

This exists because PatternMiner selects clustering hyper-parameters per ontology
by argmax of F1 against the ground truth
(`rq_0_cache/rich_eval.py:312/392`) — it consults the labels it is scored
against. The obvious concern is whether `k` was simply set to the number of
ground-truth patterns; it was not — but the whole grid was selected by the ground
truth, which is a stronger form of the same problem.

Reporting both makes the previously published figures comparable **and** honest,
and the gap is itself a result worth stating.

---

## Nothing is skipped silently

Any ontology excluded from the evaluation is written to `rq5_skipped.csv` with an
exact reason and detail:

| Reason | Meaning |
|---|---|
| `no_ontology_file` | no RDF file in the ontology directory |
| `file_too_large` | above `--max-mb` (0 = no limit) |
| `parse_error` | with the **verbatim exception type and message** |
| `no_entities` | parsed, but no entities under the WIDE definition |
| `too_many_entities` | above `--max-entities` (0 = no limit) |
| `no_ground_truth` | no pattern overlap under the requested subset |
| `embed_out_of_memory` | the embedder exhausted memory on this machine |
| `embed_error` | any other embedder failure, with its exception type |

Silent drops are how the corpus count inconsistencies arose in the first place. A
skip is data, not noise. `embed_out_of_memory` in particular means *this machine*
could not embed the ontology — the ontology remains a valid dataset member.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r use-cases/odp-extraction/requirements.txt
```

`gensim` imports `scipy.linalg.triu`, removed in scipy ≥ 1.12. The embedders call
`_common.scipy_triu_patch()` before importing gensim, so any version works.

No GPU is required; everything runs on CPU.

---

## Run it

```bash
cd use-cases/odp-extraction

# the small portions, on every gated ontology with any verified reuse
python tools/run_extraction_eval.py \
    --manifest ../../pipeline/out/MANIFEST.csv \
    --data-root <corpus> \
    --out results/small \
    --datasets github noOBO --subset reuse

# the complete-reuse subset only (the paper's original choice)
python tools/run_extraction_eval.py ... --subset complete

# DOS-DP, no caps (very large ontologies — see the warning below)
python tools/run_extraction_eval.py ... --datasets OBO \
    --max-entities 0 --max-mb 0
```

`--resume` continues an interrupted run from `rq5_per_ontology.jsonl`.

### The published methods and the reference points

```bash
# plain corpus
python tools/run_method_eval.py \
    --manifest ../../pipeline/out/MANIFEST.csv \
    --data-root <MultiSource> --out results/methods-iri \
    --datasets github noOBO --subset reuse --corpus-name plain

# IRI-anonymised twin: same population, release layout
python tools/run_method_eval.py \
    --manifest ../../pipeline/out/MANIFEST.csv \
    --data-root ../../release-iri/corpus-hashed --corpus-layout release \
    --out results/methods-iri-anon \
    --datasets github noOBO --subset reuse --corpus-name anonymised

# the embedding baselines, on the same twin
python tools/run_extraction_eval.py ... --corpus-layout release \
    --data-root ../../release-iri/corpus-hashed --out results/iri-anon

# join all four runs into the paper's table, over paired populations
python tools/make_uc1_tables.py \
    --plain results/methods-iri results/iri \
    --anon  results/methods-iri-anon results/iri-anon \
    --out stats/uc1
```

`make_uc1_tables.py` scores every plain-vs-anonymised comparison over the
**intersection** of the two populations, per dataset, because 18 GitHub
ontologies failed to anonymise and a 442-vs-424 comparison would not be paired.

### Two things the results say about the benchmark itself

1. **One entity per cluster beats every method** (F1 0.644 GitHub, 0.487 Paper &
   Repo), because the median ground-truth instance holds 2–3 entities and a
   singleton that hits its target scores `2/(|g|+1)`. Report it. It is the line
   a new method has to clear.
2. **IRI namespace grouping reaches 0.886 on the plain corpus and 0.234 on the
   anonymised twin.** Any number computed on the plain corpus has that shortcut
   available to it. Every content-based method here moves by ≤ 0.022 between the
   two, so none of them uses it — but that had to be measured.

### Outputs

| File | Content |
|---|---|
| `rq5_per_ontology.jsonl` | one row per (ontology, method, clusterer), both regimes |
| `rq5_results.csv` | aggregates per (dataset, method, clusterer) |
| `rq5_skipped.csv` | every exclusion, with reason and exact error |
| `tables/rq5_extraction.tex` | the LaTeX table, generated from the CSV |

### A warning about DOS-DP

The DOS-DP portion is not comparable in scale to the others:

| Portion | median entities per ontology |
|---|---:|
| GitHub (gated) | 68 |
| Paper & Repo (gated) | 60 |
| DOS-DP | 7,257 – 63,975 |

OWL2Vec\* and RDF2Vec build `n_walks` walks per entity, so cost and memory grow
with entity count; `zp` (63,975 entities) implies roughly 6.4 M walks. Expect
hours per ontology on CPU, and `embed_out_of_memory` on a laptop. Recovering a
ten-entity pattern by clustering 64,000 entities is a **structurally different
task**, which is why DOS-DP scores far below the other portions — report it
separately, never pooled.

---

## Choosing the population

`--subset complete` reproduces the paper's choice, but after the quality gate the
complete-reuse subset is small (single digits per portion), because the gate
removes the pattern copies that previously satisfied a 100%-exact-IRI criterion
tautologically. `--subset reuse` gives a far larger population and is the better
primary population for a benchmark claim; report complete-reuse as a strict
sub-analysis. See `CAVEATS.md`.

---

## Relation to the released dataset

Ground truth here is recomputed from the ontology and pattern files with
`pipeline/odpqual/entities.py`, and is identical to the `pairs.csv` shipped in
the release. Two conventions are not optional if you want comparable numbers:
entity sets use the **WIDE** definition, and **annotation vocabularies are
stripped from both the ontology and the pattern** before comparison. Both are
implemented in that one module, and both use cases plus the RQ statistics import
it, so no two parts of this repository can disagree about what an entity is.
