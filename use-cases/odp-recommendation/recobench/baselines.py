"""
recobench.baselines
====================

Lightweight, dependency-free recommender backends:

  * RandomRecommender       -- floor; does NOT use the reuse signal.
  * PopularityRecommender   -- rank by how often each pattern is reused in TRAIN
                               (a global prior; the simplest "uses-the-dataset"
                               method, and a famously strong recsys baseline).
  * OntologyKNNRecommender  -- user-based collaborative filtering for cold start:
                               find the TRAIN ontologies most similar to the test
                               ontology (by Jaccard overlap of their term bags),
                               then recommend the patterns those neighbours reused,
                               weighted by similarity. The headline "structural"
                               dataset method.

These three bracket Thread 1: Random (no dataset) vs Popularity / kNN-CF (dataset).
"""
from __future__ import annotations

import random as _random

from .core import Ontology, Recommender, pattern_popularity


class RandomRecommender(Recommender):
    name = "random"
    uses_dataset = False

    def __init__(self, seed: int = 7):
        self.seed = seed

    def fit(self, train, catalog):
        self.catalog = list(catalog)

    def recommend(self, onto: Ontology, top_n: int = 10):
        cat = list(self.catalog)
        _random.Random(hash(onto.name) ^ self.seed).shuffle(cat)
        return cat[:top_n]


class PopularityRecommender(Recommender):
    name = "popularity"
    uses_dataset = True

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        pop = pattern_popularity(train)
        self.ranked = sorted(catalog, key=lambda s: pop.get(s, 0), reverse=True)

    def recommend(self, onto: Ontology, top_n: int = 10):
        return self.ranked[:top_n]


class OntologyKNNRecommender(Recommender):
    """User-based CF: neighbours = most term-similar TRAIN ontologies.

    score(pattern) = sum over k nearest train ontologies that reused it of
    their similarity to the test ontology, with a small popularity tie-breaker.
    """
    name = "ontology_knn_cf"
    uses_dataset = True

    def __init__(self, k: int = 15, pop_weight: float = 0.01):
        self.k = k
        self.pop_weight = pop_weight

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.train = train
        self.pop = pattern_popularity(train)
        self._maxpop = max(self.pop.values()) if self.pop else 1

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        return inter / len(a | b) if inter else 0.0

    def recommend(self, onto: Ontology, top_n: int = 10):
        sims = []
        for tr in self.train:
            s = self._jaccard(onto.tokens, tr.tokens)
            if s > 0:
                sims.append((s, tr))
        sims.sort(key=lambda x: x[0], reverse=True)
        neighbours = sims[: self.k]

        score: dict[str, float] = {}
        for s, tr in neighbours:
            for stem in tr.gt_stems:
                if stem in self.catalog or stem in score:
                    score[stem] = score.get(stem, 0.0) + s
        # popularity tie-breaker so ties resolve sensibly, and so we still
        # return something when no neighbour overlaps.
        for stem in self.catalog:
            score[stem] = score.get(stem, 0.0) + \
                self.pop_weight * (self.pop.get(stem, 0) / self._maxpop)

        ranked = sorted(score, key=lambda s_: score[s_], reverse=True)
        return ranked[:top_n]
