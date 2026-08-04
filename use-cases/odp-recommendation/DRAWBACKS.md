# Drawbacks of ODPReco as an ODP recommender

A critical analysis of the ODPReco method (`../ODPReco`), grounded in its source
code and in the empirical results of the `recobench` benchmark in this directory.
This motivates the alternative recommenders in [recobench/](recobench/) and the
results in [results_reco/](results_reco/).

ODPReco is a 2021-era research prototype (`com.recommender.*`, "contributed by
Maleeha"). It is a reasonable first cut at the problem, but it has serious
methodological and engineering limitations that cap its usefulness on real
ground-truth data like `noOBO` and `github_dataset`.

---

## 1. Methodological drawbacks

### 1.1 No learning, and embeddings that are essentially noise
ODPReco advertises "Doc2Vec" but **trains no model that is ever reused**. For each
of 8 dimensions it builds a `ParagraphVectors` model *from scratch on a per-run
corpus and discards it* (e.g. `ParagraphVectors_Signature.java:75-86`). Worse, the
corpus is **one line per ODP** (the input ontology is `Line_0`, each of 73 ODPs is
one more line). Doc2Vec needs many sentences and repeated word contexts to learn
useful vectors; on 74 one-line "documents" with `minWordFrequency(1)` and
`layerSize(100)`, the resulting vectors are close to random.

**Empirical confirmation.** `recobench` includes `Doc2VecRecommender`, a *correct*
re-implementation (gensim, real multi-line training over the whole dataset). Even
done properly, Doc2Vec content matching scores **MAP ≈ 0.05 on noOBO — barely
above the random floor (0.03)** and far below a trivial popularity prior (0.35).
The approach is simply not suited to this task; ODPReco's even weaker one-line
variant cannot do better.

### 1.2 A fixed 73-ODP catalog → a hard recall ceiling
ODPReco can only ever recommend one of **73 hardcoded ODPs** (`resources/odp_list`,
`resources/home_files/`). Real ontologies reuse a much larger, open vocabulary of
patterns. On these datasets the catalog covers only ~39 % of github pattern types
and ~29 % of noOBO types, and **misses the most frequent ones** (`partof`,
`situation`, `collectionentity`, `participation`, `timeinterval`,
`timeindexedsituation`, `cpannotationschema`). The instance-weighted recall ceiling
is **~49 % (github) / ~33 % (noOBO)** — an upper bound ODPReco cannot exceed no
matter how good its scoring is.

**Empirical confirmation.** Running the *same* embedding model over the fixed
catalog vs a dataset-derived catalog: **MAP 0.13 (fixed) vs 0.20 (dataset)** and
**R@10 0.35 vs 0.48** on noOBO. The catalog, not the algorithm, is a primary
bottleneck.

### 1.3 The advertised score weighting is not implemented
The README and code comments claim a "Lucene 0.6 / Doc2Vec 0.4" weighting. In code
(`OrderLuceneODPsAndAssignWeight.orderAndWeight`) **no 0.6/0.4 multipliers exist**:
the final score is `lucene + normalised_doc2vec` with equal weight 1.0. Likewise
the 8 Doc2Vec dimensions plus the graph-isomorphism score are **summed with equal
weight** (`Integration_Of_Scores`), so a strong signal in one dimension is diluted
by 8 noisy ones. There is no learned or tuned combination.

### 1.4 A magic, unjustified threshold
Recommendations are everything scoring `> 0.85` after min–max normalisation
(`ODPRecommender.java:54`; the comment even says 0.8 while the literal is 0.85).
Because the score is a sum of ~9 min–max-normalised noisy components, 0.85 is
arbitrary and dataset-independent — it controls how many ODPs are returned with no
relation to relevance, and cannot be calibrated per ontology.

### 1.5 A near-useless structural signal
The "graph isomorphism" dimension (`GraphIsomorphism.java`) contributes only
`{0, 0.01, 0.02}`: 0.02 if the entire encoded graph line is byte-identical, 0.01 if
≥2 edge-tokens coincide, else 0. Against real ontologies this is almost always 0,
so the much-emphasised structural analysis adds essentially nothing.

### 1.6 No held-out evaluation, no notion of training data
ODPReco never *uses* a corpus of "ontology → reused patterns" examples — it cannot
learn that, e.g., ontologies about agents tend to reuse `agentrole`. It is purely
content-similarity to a static catalog. The biggest signal in this problem (which
patterns similar ontologies actually reused) is left on the table.

**Empirical confirmation.** The strongest `recobench` methods are exactly the ones
that exploit this: ontology-to-ontology collaborative filtering reaches **MAP ≈
0.43 / R@10 ≈ 0.63–0.70**, ~8× ODPReco-style content matching.

---

## 2. Engineering drawbacks

* **Brittle fixed-path I/O.** Every class hardcodes
  `System.getProperty("user.dir")+"/resources/..."` and **overwrites ~50 fixed
  filenames in place** every run — three score files even overwrite their own input
  files (`ontology_cq`, `disjointClasses_ontology`, `subObjectProperty_ontology`).
* **No concurrency safety.** Two runs in one working directory corrupt each other,
  so batch evaluation needs a private copy of `resources/` per ontology (this is
  exactly what `../run_odpreco_eval.py` does).
* **Hardcoded cardinality.** `73`/`74` is wired throughout (`Similarity` makes
  exactly 73 calls; `Integration`/`Normalising`/`ODPRecommender`/`StartODPReco`
  hardcode 73). The shipped catalog files are themselves inconsistent (72/73/74
  lines) — a latent off-by-one across dimensions. Swapping the catalog means
  editing and recompiling Java.
* **Global `System.setOut()` redirection** in many classes makes the code
  non-reentrant and impossible to embed cleanly in another process.
* **Buggy stopword filtering** (`StopWords_Ontology`): uses
  `stopword.contains(word)` (substring), not equality, so it removes the wrong
  tokens.
* **Hostile CLI.** `-ontdes`/`-ontcq` are declared optional but the code calls
  `.replaceAll(...)` on them unconditionally → `NullPointerException` if omitted.
* **Fragile dependency stack.** Java 8 target, `deeplearning4j`/`nd4j` 1.0.0-beta4
  (mixed with `deeplearning4j-nlp-uima` 0.9.0), Lucene 8.5.2, Stanford CoreNLP 3.8
  (declared but unused). The jar runs on JDK 11 here but the toolchain is dated and
  heavy (~minutes/ontology, dominated by JVM + native init).
* **No tests, logging config, or error recovery.** A single malformed OWL aborts
  the run.

---

## 3. What the evidence says

On `noOBO` (held-out, 30 % test split, MAP shown):

| Family | Method | MAP | Uses the dataset? |
|---|---|---|---|
| floor | random | 0.03 | no |
| ODPReco-style | Doc2Vec content match | 0.05 | content only |
| ODPReco-style | embedding over **fixed** catalog | 0.13 | no (fixed catalog) |
| content | embedding over **dataset** catalog | 0.20 | yes (catalog) |
| **dataset** | popularity prior | 0.35 | yes |
| **dataset** | embedding ontology-kNN CF | 0.37 | yes |
| **dataset** | Jaccard ontology-kNN CF | **0.43** | yes |

Two clear conclusions:
1. **Using the ground-truth reuse data is worth ~8–13× over content-only / fixed
   catalog** — the signal ODPReco ignores is the dominant one.
2. **ODPReco's core ingredients are the weakest tested** — Doc2Vec scoring is
   near-random and the fixed catalog roughly halves recall.

---

## 4. Implications for a new recommender

A better ODP recommender should:

* **Learn from the reuse matrix** (collaborative filtering over "which patterns
  similar ontologies reused"), not just content-match a static catalog.
* **Use an open, dataset-derived catalog** so it can recommend the frequent
  patterns ODPReco cannot represent.
* **Use modern pretrained text embeddings** (sentence-transformers / open LLM
  embeddings) instead of fitting Doc2Vec on one-line corpora.
* **Optionally add a generative LLM** (e.g. local Qwen via Ollama) to re-rank a
  cheap high-recall shortlist for precision and to produce human-readable
  justifications — at a fraction of ODPReco's per-ontology cost and with none of
  its fixed-path brittleness.

All four are implemented and benchmarked in [recobench/](recobench/); see
[results_reco/](results_reco/) and the generated `REPORT.md`.
