# Use case: ODP recommendation

**Question this use case answers:** *does the ODP-Reuse dataset actually make an
ontology-design-pattern recommender better?*

**Answer:** yes, and by a large margin. A recommender that learns from the
dataset's ontology→pattern reuse matrix reaches MAP 0.426, against 0.026 for a
no-data floor and 0.028 for content-only pattern matching — a paired improvement
of **+0.400 MAP** (95 % CI [0.258, 0.547], sign-test p < 0.001, winning on 19 of
23 test ontologies and losing on none).

This is the reusability demonstration for the resource: the dataset is not only
an object of analysis, it is training signal that a downstream ontology
engineering task can consume.

> **Read [`CAVEATS.md`](CAVEATS.md) before citing any number from this
> directory.** It is the citation gate. In particular: only the `noOBO`
> (Paper/Repo) portion was evaluated, on 23 test ontologies; several claims that
> appear in the raw reports do **not** survive a paired significance test; and
> the original ODPReco jar was not run at scale.

---

## What is here

```
use-cases/odp-recommendation/
├── README.md                     this file
├── CAVEATS.md                    citation gate — what may and may not be cited
├── DRAWBACKS.md                  code-level critique of the prior ODPReco system
├── requirements.txt              Python dependencies
├── pattern_catalog_map.json      GT pattern stem → ODPReco catalogue ODP
│
├── recobench/                    the benchmark (importable Python package)
│   ├── core.py                   protocols, ranking metrics, data loading
│   ├── baselines.py              random, popularity, Jaccard ontology-kNN CF
│   ├── content.py                embedding CF, embedding item retrieval, Doc2Vec
│   ├── llm.py                    local-LLM rerank / generative / pointwise
│   ├── fusion.py                 reciprocal-rank fusion, learned LTR
│   ├── finetune.py               contrastive adaptation of the encoder
│   ├── stats.py                  bootstrap CIs and paired significance tests
│   ├── run.py / run_phases.py    drivers
│   ├── report.py                 renders REPORT.md
│   └── .cache/noOBO_text.json    extracted ontology text (so a run needs no re-parse)
│
├── results/noOBO/                results with real identifiers
│   ├── comparison.csv            master table, sorted by MAP
│   ├── <backend>.json            summary + per-ontology rows
│   ├── REPORT.md                 generated narrative report
│   └── phases_report.md          improvement-phase report (see CAVEATS §3)
│
├── results_hashed/noOBO/         same results, hashed identifiers
│   ├── <backend>.json            ontology ids and pattern stems hashed
│   ├── comparison.csv            copied verbatim (holds no identifiers)
│   └── id_map.csv                crosswalk: kind, original, hashed
│
├── stats/                        tidy CSVs derived from results/ — cite these
│   ├── reco_headline_noOBO.csv   one row per recommender, with a citability flag
│   ├── reco_ci_map_noOBO.csv     bootstrap 95 % CI of mean MAP
│   ├── reco_paired_tests_noOBO.csv  paired deltas with CI + sign test
│   └── catalog_coverage.csv      fixed-catalogue coverage of the GT vocabulary
│
└── tools/
    ├── make_stats.py             regenerates everything under stats/
    ├── hash_reco_results.py      regenerates results_hashed/ from results/
    └── run_odpreco_eval.py       harness for the original ODPReco jar (see CAVEATS §3)
```

### Hashed vs plain

The dataset is released in a plain form and a hashed form. `MultiSource/hash_iris.py`
rewrites IRIs *inside* RDF files but preserves directory and file names, so a
results file keyed by ontology directory name and pattern stem still carries the
original names. `tools/hash_reco_results.py` closes that gap: it applies the same
SHA-1 + URL-safe-base64 function to the ontology identifiers and pattern stems in
the result rows, so `results_hashed/` can be consumed alongside the hashed dataset
release. Metric values are copied through untouched — only identifiers change.
`id_map.csv` is the crosswalk.

---

## The task

A **recommendation** is a ranked list of ODP stems (`dul`, `agentrole`,
`partof`, …). The **ground truth** for an ontology is the set of patterns it
actually reused — the files in its `patterns/` directory. At test time the
recommender sees only the ontology's own content (labels and local names of its
classes and properties) and never its reused patterns: a realistic cold-start
setting.

Pattern **stems** are the ground-truth key rather than IRIs, because many GitHub
pattern files carry corrupted internal IRIs. This is a deliberate robustness
choice and is why this use case is stem-keyed while the extraction benchmark
(UC on automatic ODP extraction) is IRI-keyed.

**Metrics.** Macro-averaged over test ontologies: precision@k, recall@k, f1@k,
hits@k for k ∈ {1,3,5,10}, MRR, MAP, nDCG@k, R-precision. The headline is **MAP**.

**Protocol.** `--protocol split` (70/30, seed 13) for the stored results;
`--protocol loo` (leave-one-ontology-out) is also implemented.

---

## The recommenders

| Module | Backend | Idea | Uses the dataset? |
|---|---|---|---|
| `baselines` | `random` | shuffle the catalogue — the floor | no |
| `baselines` | `popularity` | rank by reuse frequency in the training ontologies | yes |
| `baselines` | `ontology_knn_cf` | patterns of the most term-similar (Jaccard) training ontologies | yes |
| `content` | `embed_ontology_knn_cf` | the same CF with sentence-embedding similarity | yes |
| `content` | `embed_item` | embed ontology against each ODP document; catalogue is swappable between the dataset's own patterns and ODPReco's fixed 73 | yes / no |
| `content` | `doc2vec` | a correct gensim re-implementation of ODPReco's Doc2Vec idea | content only |
| `llm` | `llm_rerank` | embedding-CF shortlist, reranked by a local Qwen-2.5-7B | yes |
| `llm` | `llm_generative` | Qwen picks from a bounded ODP menu (RAG) | yes |
| `fusion` | `RRFusion` / `FusionRecommender` | reciprocal-rank fusion; learned learning-to-rank | yes |
| `finetune` | — | contrastive domain adaptation of the encoder on the reuse signal | yes |

The comparison that isolates the dataset's contribution is `embed_item` with the
**dataset-derived catalogue** vs. the **fixed 73-ODP catalogue**, holding the
embedding model constant. Note in `CAVEATS.md` §3 that this particular
comparison is directional, not significant, at n = 23.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r use-cases/odp-recommendation/requirements.txt
```

The LLM backends additionally need [Ollama](https://ollama.com) with the models
pulled:

```bash
ollama pull qwen2.5:7b-instruct
ollama pull embeddinggemma          # optional, for the Phase 1 encoder swap
```

No GPU is required. Everything except the LLM backends runs on CPU in minutes.

---

## Run it

All commands are run from this directory.

```bash
cd use-cases/odp-recommendation
```

**Reproduce the stored results** (Thread 1 + the non-LLM backends, on `noOBO`):

```bash
python -m recobench.run --dataset noOBO --protocol split --fixed-catalog
```

**Add the local-LLM backends** (slow on CPU, ~110 s per call):

```bash
python -m recobench.run --dataset noOBO --protocol split --fixed-catalog \
    --with-llm --llm-sample 23
```

**Leave-one-out instead of a single split** (recommended — see `CAVEATS.md` §4):

```bash
python -m recobench.run --dataset noOBO --protocol loo --fixed-catalog
```

**Run the GitHub portion** (not yet done; the single most valuable extension):

```bash
python -m recobench.run --dataset github --protocol split --limit 600 --fixed-catalog
```

**Improvement phases** (0 = CIs and significance, 1 = better encoder,
2 = hybrid LLM, 4 = fusion, 5 = fine-tuning):

```bash
python -m recobench.run_phases --phases 0 4 5                       # CPU-fast, no LLM
python -m recobench.run_phases --phases 1 --embspecs st:all-MiniLM-L6-v2 ollama:embeddinggemma
python -m recobench.run_phases --phases 1 2 --llm-sample 8          # Qwen backends
```

**Render the human-readable report:**

```bash
python -m recobench.report --dataset noOBO
```

**Regenerate the derived artifacts** (do this after any new run):

```bash
python tools/make_stats.py --dataset noOBO
python tools/hash_reco_results.py --in-dir results/noOBO --out-dir results_hashed/noOBO
```

### Pointing at the data

`recobench` reads the dataset from the repository's ontology directories, where
each ontology is a directory containing its OWL file and a `patterns/`
subdirectory holding the patterns it reuses — the layout under
`Ontologies_reusingODPs/`. Override the root with `--ontologies-dir` if your
copy lives elsewhere. The first run extracts and caches ontology text under
`recobench/.cache/`; `noOBO_text.json` is shipped so the stored results can be
reproduced without re-parsing.

### Running the original ODPReco jar

`tools/run_odpreco_eval.py` is a complete harness for the prior ODPReco system,
including a mode that rebuilds its fixed catalogue from the dataset's own
patterns. It requires a separate checkout of ODPReco (a ~1.8 GB Java project,
not vendored here) and a JRE. See the flag reference inside the script and the
constraints in `DRAWBACKS.md`. **No outputs from this harness are included in
this release** — see `CAVEATS.md` §3.

---

## Citing

The two supported claims, with the files backing them:

- Dataset-informed CF vs. no-data floor: ΔMAP **+0.400**, 95 % CI [0.258, 0.547],
  sign-test p < 0.001, 19/0/4 → `stats/reco_paired_tests_noOBO.csv`, row 1.
- CF vs. content-only matching: ΔMAP **+0.398**, 95 % CI [0.250, 0.555],
  sign-test p < 0.001, 18/0/5 → `stats/reco_paired_tests_noOBO.csv`, row 4.

Absolute MAP with confidence intervals: `stats/reco_ci_map_noOBO.csv`.

Everything else: check `CAVEATS.md` first.
