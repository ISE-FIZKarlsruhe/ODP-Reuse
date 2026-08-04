# CAVEATS — what may and may not be cited from this use case

The citation gate for the automatic-ODP-extraction use case, in the same form as
`PROVENANCE.md` in the PatternMiner repository and `CAVEATS.md` in the
recommendation use case. **If a claim is not listed as citable below, it must not
appear in a paper, table or slide.**

---

## 1. The single most important constraint

**Every previously published F1 for this task is an oracle upper bound, not an
achievable score.**

PatternMiner selects the clustering hyper-parameters per ontology — `k` for
K-Means, `min_cluster_size` for HDBSCAN, linkage, fuzziness — by argmax of macro
entity-F1 **computed against the ground truth**
(`rq_0_cache/rich_eval.py:312/392`, `rq_3_clustering/src/clusterers.py:72`). The
selection step consults the labels the method is then scored on. There is no
unsupervised model-selection path anywhere in that codebase except `rq_4`'s
silhouette sweep.

The obvious concern is whether `k` was simply pinned to `|GT|`. It was not --
but the entire grid was chosen by the ground truth, which is a stronger form of
the same problem.

Consequences for writing:

- Any `oracle_*` figure must be labelled an upper bound. Acceptable phrasing:
  *"with clustering hyper-parameters selected per ontology against the ground
  truth (an upper bound)"*.
- The `unsup_*` figures are the achievable numbers and should carry the argument.
- `oracle_gap_f1` should be reported, because quantifying the gap is a
  contribution in itself.
- **Do not** compare a published `oracle` figure from prior work against an
  `unsup` figure here as if they were the same measurement.

---

## 2. Population caveats

| Caveat | Detail |
|---|---|
| **The complete-reuse subset is now small** | The quality gate removes ontologies that are themselves copies of patterns (14 of the previously published 49) and byte-identical duplicates. What remains per portion is single-digit. A benchmark claim cannot rest on it; use `--subset reuse` as the primary population and report complete-reuse as a strict sub-analysis. |
| **DOS-DP is not comparable** | Median entities per ontology: GitHub 68, Paper & Repo 60, DOS-DP 7,257–63,975. Recovering a ten-entity pattern from a 64,000-entity ontology is a different task. Never pool DOS-DP with the others, and never report a single averaged F1 across all three portions. |
| **The corpus is a snapshot** | Discovery queried live APIs. Re-harvesting will not reproduce it; use the archived deposition. |
| **Two ontologies exceed practical embedding limits** | `mondo-simple` and `upheno` are >200 MB. They are valid dataset members and are retained in the corpus, but embedding them may fail with `embed_out_of_memory` on commodity hardware. Any such failure is recorded in `rq5_skipped.csv` — check that file before reporting an *n*. |

---

## 3. Report the denominators, always

`rq5_results.csv` carries `n_ontologies` per (dataset, method, clusterer). These
can differ between methods, because an embedder may fail on an ontology another
handles. **A mean F1 without its denominator is not citable.** If two methods are
compared, either restrict to their common ontologies or state both *n* values.

This is the same defect PatternMiner's `leaderboard_fair.csv` guards against with
its `n_ont_all` / `n_ont_common` split — where `n_ont_all` ranges 394–580 across
methods, making `mean_f1_all` non-comparable and only `mean_f1_common` (n=361)
meaningful.

---

## 4. What must never be claimed

| Claim | Why not |
|---|---|
| *"OWL2Vec\* is the best embedding method for ODP extraction."* | This was concluded from a two-method comparison. On PatternMiner's PROVENANCE-cleared 361-ontology common set, OWL2Vec\* ranks **11 of 45** and RDF2Vec **17 of 45**; region-based methods (`elbe`, `el_embeddings`, `box2el`) lead. The two-method conclusion does not survive a wider field. |
| Any absolute F1 from `PatternMiner/main_results.csv` | It contains `learned_mlp`, which trains on the ground truth of the ontology it is scored on — an oracle, not a method. PROVENANCE marks it not citable. |
| A DOS-DP number produced by the *old* pipeline | DOS-DP is entirely absent from `PatternMiner/aggregates/` (zero rows; no OBO dataset label in any CSV). The paper's DOS-DP row came from a retired run and cannot be regenerated from current artifacts. Re-run it or drop it — do not carry the old number forward. |
| A pooled figure across all three portions | See DOS-DP above. |

---

## 5. Statistical practice

The current outputs are point estimates. Before making a comparative claim:

1. Compute a **paired** per-ontology difference, not a difference of means — the
   ontologies vary enormously in difficulty. `recobench.stats.paired_delta` in the
   recommendation use case does this and can be reused directly.
2. Report a bootstrap CI and an exact sign test, and call a difference
   significant only when the CI excludes zero **and** the sign test agrees.
3. Correct for multiple comparisons (Holm–Bonferroni) once more than a handful of
   pairs are reported.

For reference, in the recommendation use case three of five plausible-looking
point-estimate gaps did **not** survive this treatment. Expect the same here.

---

## 6. Reproducibility

Seeds are fixed: embedders `seed=42`, K-Means `random_state=42`. The manifest is
the population definition — record which `MANIFEST.csv` a result came from, since
re-gating with different thresholds changes the population.

> **CORRECTION (2026-08-04).** This section previously claimed that "re-running
> on the same corpus and manifest reproduces the same numbers". **That is true
> in aggregate and false per ontology.** `gensim.models.Word2Vec` defaults to
> `hashfxn=hash`, and CPython salts string hashing per process, so the learned
> vectors depend on `PYTHONHASHSEED` regardless of `seed=42` and `workers=1`.
> The published run did not pin it.
>
> Measured consequence, comparing runs under different hash seeds:
>
> | | |
> |---|---|
> | aggregate macro-averaged F₁ per cell | reproduces to within **0.007** |
> | ontologies whose selected config changes | **31–66%** |
> | ontologies moving more than 0.10 F₁ | **3–31%** |
> | mean change in chosen cluster count | 0.5–3.1 clusters |
>
> **What this licenses.** Aggregate claims — every figure in the paper's UC1
> table — are safe. Any claim about an individual ontology, any per-ontology
> paired test, and any ranking built from per-ontology wins are **not**
> reproducible as the pipeline currently stands. This is why UC1 reports no
> paired test over ontologies while UC2 does.
>
> **Fix:** pass a deterministic `hashfxn` to `Word2Vec` (e.g. a `zlib.crc32`
> of the token) or pin and record `PYTHONHASHSEED`. Also persist
> `n_predicted`, `n_noise`, `cluster_sizes` and `silhouette` per record — their
> absence meant the cluster-count and split/merge statistics had to be
> recovered by re-running rather than read off the release.
