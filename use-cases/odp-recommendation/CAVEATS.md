# CAVEATS — what may and may not be cited from this use case

This file is the **citation gate** for the ODP recommendation use case. If a
claim is not listed as citable below, it must not appear in a paper, table or
slide. It follows the same convention as `PROVENANCE.md` in the PatternMiner
repository.

Every number below was recomputed from the stored per-ontology result files by
`tools/make_stats.py`; none is transcribed from a prose report.

---

## 1. Scope limits that apply to everything here

| Limit | Detail |
|---|---|
| **One dataset only** | Only the `noOBO` (Paper/Repo) portion was evaluated. `github_dataset` was **not run**. No claim may be generalised to the GitHub portion. |
| **Small test set** | 70/30 split, seed 13 → **23 test ontologies**, mean 2.3 ground-truth patterns each. |
| **Single split** | One split, one seed. Not repeated splits, not leave-one-out (`--protocol loo` exists but the stored results are the split protocol). |
| **LLM rows are a sample** | The Qwen backends were scored on **8** ontologies (Phase 2 in-context CF on **5**), not 23. LLM rows are never comparable to full-roster rows by raw MAP. |
| **The ODPReco JAR was never run at scale** | See §3. There are **no** mode-A (real jar) results in this release. |

Consequence: this use case demonstrates **that the dataset is usable and
useful**, not that any particular recommender is best. Write it that way.

---

## 2. Claims that ARE supported

Both are paired, per-ontology comparisons on the same 23 test ontologies, with a
95 % percentile-bootstrap CI (5000 resamples, seed 0) and an exact two-sided
sign test. Source: `stats/reco_paired_tests_noOBO.csv`.

| Claim | ΔMAP | 95 % CI | sign-test p | W/L/T |
|---|---|---|---|---|
| **A dataset-informed recommender beats the no-data floor.** Ontology-kNN collaborative filtering over the ODP-Reuse reuse matrix vs. random ranking. | **+0.400** | [0.258, 0.547] | < 0.001 | 19/0/4 |
| **Collaborative filtering over the reuse matrix beats content-only pattern matching.** Same CF vs. a correct gensim re-implementation of ODPReco's Doc2Vec scoring idea. | **+0.398** | [0.250, 0.555] | < 0.001 | 18/0/5 |

These are the two claims the paper should rest on. Both are large, both have CIs
that exclude zero by a wide margin, and both lose on **zero** ontologies.

Absolute values for context (`stats/reco_ci_map_noOBO.csv`), MAP with 95 % CI:

| Recommender | Uses ODP-Reuse data | MAP | 95 % CI |
|---|---|---|---|
| ontology_knn_cf (Jaccard) | yes | 0.426 | [0.286, 0.572] |
| embed_ontology_knn_cf (MiniLM) | yes | 0.359 | [0.243, 0.482] |
| popularity | yes | 0.353 | [0.240, 0.474] |
| embed_item, dataset catalogue | yes | 0.200 | [0.109, 0.307] |
| embed_item, ODPReco fixed catalogue | no | 0.129 | [0.053, 0.233] |
| doc2vec (re-implementation) | content only | 0.028 | [0.002, 0.074] |
| random | no | 0.026 | [0.006, 0.052] |

---

## 3. Claims that are NOT supported — do not make them

| Claim as it appears in the source reports | Status | Why |
|---|---|---|
| *"A dataset-derived catalogue beats ODPReco's fixed 73-ODP catalogue by +55 % MAP."* | **NOT SIGNIFICANT** | Paired ΔMAP **+0.071**, CI **[−0.089, +0.218]**, sign-test p = **0.189**, 14/7/2. The CI includes zero. The +55 % is a ratio of two point estimates whose CIs overlap heavily (0.129 vs 0.200). State it as *directional* or drop it. |
| *"CF beats the popularity prior."* | **NOT SIGNIFICANT** | ΔMAP +0.073, CI [−0.018, +0.168], p = 0.302, 10/5/8. A popularity prior is a genuinely strong baseline here — say so; it is an honest and interesting result. |
| *"LLM reranking improves recommendation."* | **NOT SUPPORTED** | ΔMAP **−0.024** vs its own retrieval base, CI [−0.202, +0.141], p = 1.000, 3/4/1, and only n = 8. The local LLM does **not** beat collaborative filtering. |
| *"embeddinggemma CF is the best method (MAP 0.458)."* | **NOT RE-DERIVABLE** | Reported in `results/noOBO/phases_report.md` but the per-ontology JSON for that run is **not in this release**, so no CI can be computed and `make_stats.py` cannot verify it. Cite only if the run is repeated and its JSON archived. |
| *"In-context CF is the strongest LLM method (MAP 0.559)."* | **ANECDOTAL** | n = 5. Report as an observation, never as a result. |
| Any number attributed to **ODPReco itself** | **UNAVAILABLE** | See below. |

### The ODPReco jar was not run at scale

`tools/run_odpreco_eval.py` implements a full harness for running the real
ODPReco jar (mode A) and there are no stored outputs from it — the
`results/results_<dataset>/` tree that harness writes does **not** exist in this
release. The row labelled `doc2vec` is **a correct gensim re-implementation of
ODPReco's Doc2Vec scoring idea**, written for this benchmark; it is *not*
ODPReco's output, and it is deliberately a *stronger* implementation than
ODPReco's (which re-fits Doc2Vec on one-line corpora per run).

Therefore: **do not write "ODPReco scores MAP 0.03".** Write "a faithful
re-implementation of ODPReco's content-matching approach scores MAP 0.03; the
original implementation is expected to be no stronger, for the reasons set out
in `DRAWBACKS.md`, but was not run at scale here."

The `DRAWBACKS.md` critique of ODPReco's *design* is grounded in its source code
and is citable as a code-level analysis. Its empirical rows inherit the caveats
above.

### Catalogue coverage numbers

`stats/catalog_coverage.csv` is derived from `pattern_catalog_map.json`, which
is a **union across both datasets**: 38 of 74 observed pattern stems (51 %) map
to an ODPReco catalogue entry. This is **not** a per-dataset figure.

The per-dataset ceilings quoted in `recobench/README.md` — 39 % of github stems,
29 % of noOBO stems; instance-weighted recall ceiling ~49 % github / ~33 % noOBO
— come from the original analysis and are **not recomputed by
`make_stats.py`**. Recompute them before citing, or attribute them to the
harness README.

---

## 4. How to strengthen this use case before submission

In rough order of value per unit effort:

1. **Run `github_dataset`.** The single biggest weakness is that only 23 test
   ontologies were evaluated. The GitHub portion has hundreds. Command is in
   `README.md` §Run it.
2. **Use repeated splits or leave-one-out** instead of one seed-13 split, and
   report the CI across splits. `--protocol loo` already exists.
3. **Archive the per-ontology JSON for the phase runs** (embeddinggemma, RRF,
   fine-tuning) so their numbers become verifiable rather than report-only.
4. **Run the real ODPReco jar** on at least a sample, so the comparison is
   against the actual prior system rather than a re-implementation.
5. **Correct for multiple comparisons** once more than a handful of paired tests
   are reported (Holm–Bonferroni over the family).
