# Use cases

Two demonstrations that the dataset is *consumable*, not merely descriptive. The
numbering matches the manuscript: UC1 uses the corpus as **ground truth**, UC2
uses it as **training signal**.

## UC1 · The dataset as ground truth

Recover ODP instances from an ontology with no knowledge of which patterns it
reuses. Embed, cluster, match clusters to ground truth by Hungarian assignment.

Every configuration is reported **twice** — once with hyper-parameters chosen
against the ground truth (an upper bound, and what prior work reported), once
chosen by silhouette without ever consulting it.

| Portion | Model | Clustering | n | Oracle F₁ | Unsupervised F₁ | Gap |
|---|---|---|---:|---:|---:|---:|
| GitHub | OWL2Vec* | K-Means | 442 | 0.453 | 0.321 | +0.132 |
| GitHub | OWL2Vec* | HDBSCAN | 442 | 0.358 | 0.243 | +0.115 |
| GitHub | RDF2Vec | K-Means | 442 | 0.468 | 0.300 | +0.167 |
| GitHub | RDF2Vec | HDBSCAN | 442 | 0.348 | 0.259 | +0.089 |
| Paper &amp; Repository | OWL2Vec* | K-Means | 51 | 0.541 | 0.448 | +0.092 |
| Paper &amp; Repository | OWL2Vec* | HDBSCAN | 51 | 0.365 | 0.324 | +0.041 |
| Paper &amp; Repository | RDF2Vec | K-Means | 51 | 0.554 | 0.435 | +0.119 |
| Paper &amp; Repository | RDF2Vec | HDBSCAN | 51 | 0.365 | 0.305 | +0.060 |

<div id="odp-uc1" class="odp-viz" data-src="data/uc1.json"></div>

The gap is the contribution of oracle model selection. Neither number is wrong;
only one of them is achievable without already knowing the answer.

## UC2 · The dataset as training signal

Given an ontology under development, which ODPs should its author reuse? The
prior system, ODPReco, matches content against a fixed 73-ODP catalogue and uses
no corpus of confirmed reuse at all — it cannot learn that ontologies about
agents tend to reuse `agentrole`, because it has no examples.

Each row is a **paired** per-ontology comparison on the same held-out ontologies,
so no difference is confounded by which ontologies happen to be easy. Only
gate-passing ontologies are used. Row 1 is the measurement of what this dataset
is worth: the best reuse-informed recommender against the best that pattern
documentation alone supports.

**GitHub portion, quality-gated**

| Comparison | n | Δ MAP | 95% CI | W/L/T | Significant |
|---|---:|---:|---|---|---|
| dataset-informed CF vs no-dataset floor | 89 | **+0.822** | [+0.754, +0.884] | 85/1/3 | **yes** |
| CF vs popularity prior | 89 | **+0.201** | [+0.137, +0.271] | 31/2/56 | **yes** |
| CF vs Doc2Vec content matching | 89 | **+0.805** | [+0.729, +0.874] | 83/3/3 | **yes** |

**Paper &amp; Repository portion — underpowered, for completeness**

| Comparison | n | Δ MAP | 95% CI | W/L/T | Significant |
|---|---:|---:|---|---|---|
| dataset-informed CF vs no-dataset floor | 14 | **+0.275** | [+0.037, +0.525] | 10/4/0 | no |
| CF vs popularity prior | 14 | **-0.008** | [-0.066, +0.050] | 3/2/9 | no |
| CF vs Doc2Vec content matching | 14 | **+0.272** | [+0.016, +0.546] | 7/4/3 | no |

<div id="odp-uc2" class="odp-viz" data-src="data/uc2.json"></div>

!!! note "What is *not* claimed"
    A faithful re-implementation of ODPReco's Doc2Vec approach stands in for the
    original, which was not run at scale — so the comparison is to the *approach*,
    not to published ODPReco numbers. On the Paper &amp; Repository portion
    ($n = 14$) nothing reaches significance and no row there should be read as a
    contrary finding.
