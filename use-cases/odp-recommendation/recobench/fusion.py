"""
recobench.fusion  --  Phase 4: learning-to-rank fusion.

The single methods each expose one signal: Jaccard CF, embedding CF, embedding
item-retrieval, and popularity. This module combines them. For every
(ontology, candidate pattern) pair a small feature vector is built and a
gradient-boosted model is trained to predict relevance (whether the ontology
reused the pattern). Candidates are then ranked by the model's score.

Leakage control: the collaborative features of a TRAINING ontology are computed
with that ontology excluded from its own neighbour pool (leave-one-out within
train); a TEST ontology is scored against the full training pool. Only training
ontologies contribute labels to the ranker.

Implemented as a `Recommender`, so `evaluate_recommender` / `evaluate_loo` score
it exactly like every other backend. CPU-only, no LLM calls; a full noOBO run is
seconds.
"""
from __future__ import annotations

import logging

import numpy as np

from .core import Ontology, Recommender, pattern_popularity
from .content import Embedder, build_pattern_docs, _cos

log = logging.getLogger("recobench.fusion")

FEATURES = ["jaccard_cf", "embed_cf", "embed_item", "popularity",
            "jaccard_top", "embed_top"]


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    return i / len(a | b) if i else 0.0


class RRFusion(Recommender):
    """Unsupervised Reciprocal Rank Fusion of several base recommenders.

    score(pattern) = sum_m 1/(k0 + rank_m(pattern)); combining ranked lists
    avoids the incomparable score scales and the overfitting that sink a learned
    fusion on a tiny training set. No parameters are learned from labels."""
    name = "fusion_rrf"
    uses_dataset = True

    def __init__(self, make_bases, k0: int = 60, label: str | None = None):
        self.make_bases = make_bases
        self.k0 = k0
        if label:
            self.name = label

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.bases = [m() for m in self.make_bases]
        for b in self.bases:
            b.fit(train, catalog)

    def recommend(self, onto, top_n: int = 10):
        n = len(self.catalog)
        score: dict[str, float] = {}
        for b in self.bases:
            ranked = b.recommend(onto, top_n=n)
            for rank, stem in enumerate(ranked):
                score[stem] = score.get(stem, 0.0) + 1.0 / (self.k0 + rank + 1)
        return sorted(score, key=lambda s: score[s], reverse=True)[:top_n]


class FusionRecommender(Recommender):
    name = "fusion_ltr"
    uses_dataset = True

    def __init__(self, embspec: str = "st:all-MiniLM-L6-v2", k: int = 15,
                 model: str = "gb", label: str | None = None):
        self.emb = Embedder(embspec)
        self.k = k
        self.model_kind = model
        if label:
            self.name = label

    # ---- feature construction -------------------------------------------
    def _cf_scores(self, q_tokens: set, q_vec: np.ndarray,
                   pool: list[Ontology], pool_vecs: np.ndarray) -> tuple[dict, dict]:
        """Return {stem: jaccard_cf}, {stem: embed_cf} for one query ontology
        against a neighbour pool (query already excluded from pool)."""
        jac = np.array([_jaccard(q_tokens, o.tokens) for o in pool])
        cos = _cos(q_vec[None, :], pool_vecs)[0] if len(pool) else np.zeros(0)
        j_j = {}
        j_e = {}
        for sims, out in ((jac, j_j), (cos, j_e)):
            order = np.argsort(-sims)[: self.k]
            for idx in order:
                s = float(sims[idx])
                if s <= 0:
                    continue
                for stem in pool[idx].gt_stems:
                    out[stem] = out.get(stem, 0.0) + s
        return j_j, j_e

    def _feature_rows(self, query: Ontology, q_vec: np.ndarray,
                      pool: list[Ontology], pool_vecs: np.ndarray) -> dict:
        """{stem: feature-vector} over the catalog for one query ontology.

        Each raw signal is additionally rank-normalised WITHIN this ontology's
        candidate set (percentile in [0,1]); ranking models compare candidates
        of the same ontology, so within-ontology scale is what matters, and the
        rank transform stops a dense signal (embed_item) from dominating a sparse
        but strong one (CF) purely by having more distinct values."""
        jac_cf, emb_cf = self._cf_scores(query.tokens, q_vec, pool, pool_vecs)
        item_cos = _cos(q_vec[None, :], self.item_mat)[0]  # ontology vs pattern docs
        raw = np.zeros((len(self.catalog), 4), dtype=float)
        for i, stem in enumerate(self.catalog):
            raw[i, 0] = jac_cf.get(stem, 0.0)
            raw[i, 1] = emb_cf.get(stem, 0.0)
            raw[i, 2] = float(item_cos[i]) if i < len(item_cos) else 0.0
            raw[i, 3] = self.pop.get(stem, 0) / self._maxpop
        # per-ontology rank normalisation of each column -> [0,1]
        ranked = np.zeros_like(raw)
        n = len(self.catalog)
        for c in range(4):
            order = np.argsort(np.argsort(raw[:, c]))
            ranked[:, c] = order / max(n - 1, 1)
        feats = {}
        for i, stem in enumerate(self.catalog):
            feats[stem] = list(raw[i]) + [ranked[i, 0], ranked[i, 1]]
        return feats

    # ---- Recommender API ------------------------------------------------
    def fit(self, train, catalog):
        self.train = train
        self.catalog = list(catalog)
        self.pop = pattern_popularity(train)
        self._maxpop = max(self.pop.values(), default=1) or 1
        self.train_vecs = self.emb.embed([o.text or o.name for o in train])
        # pattern-document embeddings aligned to self.catalog
        pdocs = build_pattern_docs(train)
        self.item_mat = self.emb.embed([pdocs.get(s, s) for s in self.catalog])

        # training rows: leave-one-out within train
        X, y, groups = [], [], []
        for i, o in enumerate(train):
            pool = train[:i] + train[i + 1:]
            pool_vecs = np.vstack([self.train_vecs[:i], self.train_vecs[i + 1:]]) \
                if len(train) > 1 else self.train_vecs
            feats = self._feature_rows(o, self.train_vecs[i], pool, pool_vecs)
            groups.append(len(feats))
            for stem, fv in feats.items():
                X.append(fv)
                y.append(1 if stem in o.gt_stems else 0)
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=int)
        self._groups = groups  # candidate rows per training ontology

        if self.model_kind == "pairwise":
            self._fit_pairwise(X, y, groups)
        elif self.model_kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.clf.fit(X, y)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=1)
            self.clf.fit(X, y)
        log.info("fusion(%s) trained on %d rows (%d positive)",
                 self.model_kind, len(y), int(y.sum()))

    def _fit_pairwise(self, X, y, groups):
        """RankNet-style linear LTR: learn weights w so that within each ontology
        a relevant candidate outranks an irrelevant one. Train a logistic model on
        difference vectors f(pos) - f(neg)."""
        from sklearn.linear_model import LogisticRegression
        diffs, labels = [], []
        start = 0
        rng = np.random.default_rng(1)
        for g in groups:
            block = slice(start, start + g)
            start += g
            yb = y[block]
            Xb = X[block]
            pos = np.where(yb == 1)[0]
            neg = np.where(yb == 0)[0]
            if len(pos) == 0 or len(neg) == 0:
                continue
            # cap negatives per positive to keep it balanced/fast
            for p in pos:
                chosen = rng.choice(neg, size=min(20, len(neg)), replace=False)
                for q in chosen:
                    diffs.append(Xb[p] - Xb[q]); labels.append(1)
                    diffs.append(Xb[q] - Xb[p]); labels.append(0)
        self.clf = LogisticRegression(max_iter=2000, fit_intercept=False)
        self.clf.fit(np.array(diffs), np.array(labels))

    def feature_importance(self) -> dict:
        imp = getattr(self.clf, "feature_importances_", None)
        if imp is None:
            imp = getattr(self.clf, "coef_", np.zeros((1, len(FEATURES))))[0]
        return {f: round(float(w), 4) for f, w in zip(FEATURES, imp)}

    def recommend(self, onto: Ontology, top_n: int = 10):
        q_vec = self.emb.embed([onto.text or onto.name])[0]
        feats = self._feature_rows(onto, q_vec, self.train, self.train_vecs)
        stems = list(feats.keys())
        X = np.array([feats[s] for s in stems], dtype=float)
        if self.model_kind == "pairwise":
            scores = X @ self.clf.coef_[0]
        elif hasattr(self.clf, "predict_proba"):
            scores = self.clf.predict_proba(X)[:, 1]
        else:
            scores = self.clf.decision_function(X)
        order = np.argsort(-scores)
        return [stems[i] for i in order[:top_n]]
