# recobench — a held-out ODP-recommender benchmark

`recobench` answers the two questions in this use-case with one evaluation
harness and swappable recommender backends:

* **Thread 1 — does using the ground-truth datasets help?** Compare recommenders
  that use the `noOBO` / `github_dataset` reuse data against ones that do not
  (random; ODPReco's fixed 73-ODP catalog).
* **Thread 2 — a new open-source-LLM ODP recommender.** Embedding retrieval,
  embedding-based collaborative filtering, and local-LLM (Qwen via Ollama)
  rerank / generation, all scored on the same protocol.

It reuses the repo's canonical data loaders (`rq_0_cache/shared_utils.py`) so the
ground truth is identical to every other baseline.

## Problem setup

A **recommendation** is a ranked list of ODP **stems** (e.g. `dul`, `agentrole`,
`partof`). The **ground truth** for an ontology is the set of pattern stems it
actually reused — the filenames in its `patterns/` directory (stems are the only
trustworthy GT key; github pattern files carry corrupted internal IRIs). At test
time a recommender sees only the ontology's content (labels + local names of its
classes/properties), never its reused patterns — i.e. realistic cold start.

## Protocols

* `--protocol loo` — leave-one-ontology-out (default; best for small `noOBO`).
* `--protocol split` — train/test split (`--test-frac`, `--seed`; use for the
  larger `github_dataset` and for the LLM/embedding backends).

The item universe (catalog) is the set of pattern stems seen in the data; for the
fixed-vs-dataset study it is swapped for ODPReco's 73-ODP catalog.

## Metrics (`core.ranking_metrics`)

Macro-averaged over test ontologies: `precision@k`, `recall@k`, `f1@k`,
`hits@k` (k ∈ {1,3,5,10}), `MRR`, `MAP`, `nDCG@k`, `R-precision`. The headline is
**MAP**.

## Backends

| module | backend | idea | uses data |
|---|---|---|---|
| `baselines` | `random` | shuffle catalog (floor) | no |
| `baselines` | `popularity` | rank by train reuse frequency | yes |
| `baselines` | `ontology_knn_cf` | CF: patterns of the most term-similar (Jaccard) train ontologies | yes |
| `content` | `embed_ontology_knn_cf` | CF with sentence-embedding ontology similarity | yes |
| `content` | `embed_item` | embed ontology vs each ODP doc, cosine (catalog-swappable: dataset **or** ODPReco's fixed catalog) | yes / no |
| `content` | `doc2vec` | a *correct* gensim re-implementation of ODPReco's Doc2Vec idea | yes |
| `llm` | `llm_rerank` | embedding-CF shortlist → Qwen reranks | yes |
| `llm` | `llm_generative` | Qwen picks from a bounded ODP menu (RAG) | yes |
| `llm` | `llm_pointwise` | Qwen scores each shortlisted ODP 0–3 (Phase 1) | yes |
| `llm` | `llm_fewshot_rerank` | rerank with reuse examples of similar ontologies = in-context CF (Phase 2) | yes |
| `llm` | `SelfConsistentRecommender` | aggregate several stochastic runs by Borda count (Phase 2) | yes |
| `fusion` | `RRFusion` | unsupervised reciprocal-rank fusion of several backends (Phase 4) | yes |
| `fusion` | `FusionRecommender` | learning-to-rank over CF/embedding/popularity features (Phase 4) | yes |

Embeddings come from `--embspec st:<model>` (sentence-transformers, default
`all-MiniLM-L6-v2`, CPU, fast) or `--embspec ollama:<model>` (e.g.
`embeddinggemma`). The LLM backends use local Ollama (`qwen2.5:7b-instruct`).

## Improvement phases (`run_phases`, `stats`, `fusion`, `finetune`)

The `run_phases` driver evaluates five improvement directions and writes
`results_reco/phases_<dataset>/phases_report.md`:

| phase | module | what it does |
|---|---|---|
| 0 trustworthy eval | `stats` | bootstrap confidence intervals + paired significance tests (bootstrap + sign test) over the stored roster — no new runs needed |
| 1 better embedder | `content` (`--embspecs`) + `llm.LLMPointwiseScorer` | swap the embedding model; LLM-as-scorer |
| 2 hybrid LLM | `llm.LLMFewShotRerank`, `SelfConsistentRecommender` | in-context CF (few-shot with similar ontologies' reuse), self-consistency |
| 4 fusion | `fusion.RRFusion` / `FusionRecommender` | combine signals (unsupervised RRF; learned LTR) |
| 5 fine-tuning | `finetune` | contrastive domain-adaptation of the embedder on the reuse signal (train ontologies only), then re-evaluate the embedding backends |

```bash
python -m recobench.run_phases --phases 0 4 5              # CPU-fast, no LLM
python -m recobench.run_phases --phases 1 --embspecs st:all-MiniLM-L6-v2 ollama:embeddinggemma
python -m recobench.run_phases --phases 1 2 --llm-sample 8 # Qwen backends (slow, CPU)
```

Note: the Ollama models (`qwen2.5:7b-instruct`, `embeddinggemma`) must be pulled
(`ollama pull ...`); Ollama evicts models, so re-pull if `ollama list` is empty.

## Run it

```bash
cd ODPReco_useCase

# Thread 1 + non-LLM Thread 2, leave-one-out on noOBO:
python -m recobench.run --dataset noOBO --protocol loo --fixed-catalog

# add the Qwen LLM backends (split protocol, sampled — CPU cost):
python -m recobench.run --dataset noOBO --protocol split --fixed-catalog \
    --with-llm --llm-sample 23

# github (large): use a split and a slice
python -m recobench.run --dataset github --protocol split --limit 600 --fixed-catalog

# render the human-readable report:
python -m recobench.report --dataset noOBO
```

Outputs land in `results_reco/<dataset>/`: one `<backend>.json` per method
(summary + per-ontology rows), `comparison.csv` (master table, sorted by MAP),
and `REPORT.md`. The first run extracts and caches ontology text under
`recobench/.cache/`; later runs are fast.

## Dependencies

All already present in this environment: `numpy`, `scipy`, `rdflib`,
`sentence-transformers` + `torch`, `gensim` (Doc2Vec; a scipy-`triu` shim is
applied automatically), and `ollama` with `qwen2.5:7b-instruct` + `embeddinggemma`
pulled. No GPU required; no network needed once models are cached.

See [`../DRAWBACKS.md`](../DRAWBACKS.md) for the ODPReco critique and
`results_reco/<dataset>/REPORT.md` for the numbers.
