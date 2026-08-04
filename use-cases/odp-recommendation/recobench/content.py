"""
recobench.content
==================

Content / embedding recommender backends (Thread 2 core + Thread 1 method):

  * EmbeddingItemRecommender  -- embed the test ontology and each catalog item's
                                 DOCUMENT, recommend by cosine. Catalog-swappable:
                                 with the dataset catalog each item IS a pattern
                                 stem; with ODPReco's FIXED 73-ODP catalog each
                                 item is an ODP (doc = its odps_description.txt
                                 text) that projects to a GT stem via the map.
                                 -> this single backend yields the fixed-vs-dataset
                                 catalog comparison for an LLM-embedding method.
  * EmbeddingOntologyKNN      -- collaborative filtering where ontology-ontology
                                 similarity is cosine of sentence embeddings
                                 (usually stronger than the Jaccard kNN baseline).
  * Doc2VecRecommender        -- a FAITHFUL but correct re-implementation of
                                 ODPReco's "Doc2Vec content similarity" idea using
                                 gensim, trained on the multi-line ontology+pattern
                                 corpus (ODPReco fits on 1-line corpora -> noise;
                                 this is the controlled, fixed dependency).

Embeddings come from an open-source model via one of two providers:
  - 'st:<model>'      sentence-transformers (default 'all-MiniLM-L6-v2'), CPU.
  - 'ollama:<model>'  local Ollama embeddings (e.g. 'embeddinggemma').
Both are cached per process. Heavy deps are imported lazily.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .core import Ontology, Recommender, tokenize, _localname

log = logging.getLogger("recobench.content")


# ----------------------------------------------------------------------
# Embedding providers
# ----------------------------------------------------------------------

class Embedder:
    """Unified text->vector with a tiny in-process cache. spec = 'st:model' or
    'ollama:model'. Default sentence-transformers all-MiniLM-L6-v2 (CPU, fast)."""

    def __init__(self, spec: str = "st:all-MiniLM-L6-v2"):
        self.spec = spec
        self.kind, self.model_name = spec.split(":", 1)
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    def _ensure(self):
        if self._model is not None or self.kind == "ollama":
            return
        from sentence_transformers import SentenceTransformer
        log.info("loading sentence-transformer %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def _embed_one_ollama(self, text: str) -> np.ndarray:
        import ollama
        r = ollama.embeddings(model=self.model_name, prompt=text[:8000])
        return np.asarray(r["embedding"], dtype=np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in texts if t not in self._cache]
        if todo:
            if self.kind == "st":
                self._ensure()
                vecs = self._model.encode(todo, normalize_embeddings=True,
                                          show_progress_bar=False,
                                          batch_size=64)
                for t, v in zip(todo, vecs):
                    self._cache[t] = np.asarray(v, dtype=np.float32)
            else:  # ollama
                for t in todo:
                    v = self._embed_one_ollama(t)
                    n = np.linalg.norm(v)
                    self._cache[t] = v / n if n else v
        mat = np.vstack([self._cache[t] for t in texts])
        return mat


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """rows of a (n,d) vs rows of b (m,d), assuming both L2-normalised -> (n,m)."""
    return a @ b.T


# ----------------------------------------------------------------------
# Pattern documents (the dataset-derived item catalog)
# ----------------------------------------------------------------------

def build_pattern_docs(train: list[Ontology], max_files_per_stem: int = 6
                       ) -> dict[str, str]:
    """{stem: document text} aggregated from TRAIN ontologies' pattern files.

    A pattern's document = its stem words + labels/comments/local-names read
    from a few example pattern OWL files (rdflib). This is what the embedding
    item-retriever matches the ontology against. Falls back to stem words only.
    """
    from collections import defaultdict
    try:
        from rdflib import Graph, RDFS, URIRef
    except Exception:
        Graph = None

    files_by_stem: dict[str, list[Path]] = defaultdict(list)
    for o in train:
        pdir = o.owl_path.parent / "patterns"
        if not pdir.is_dir():
            continue
        for f in pdir.iterdir():
            if f.is_file() and f.suffix.lower() in (".owl", ".ttl", ".rdf", ".xml"):
                stem = f.stem.strip().lower()
                if len(files_by_stem[stem]) < max_files_per_stem:
                    files_by_stem[stem].append(f)

    docs: dict[str, str] = {}
    for stem, files in files_by_stem.items():
        words = set(tokenize(stem))
        if Graph is not None:
            for f in files:
                g = Graph()
                ok = False
                for fmt in ((("turtle",) if f.suffix.lower() == ".ttl"
                             else ("xml",)) + ("xml", "turtle")):
                    try:
                        g.parse(str(f), format=fmt); ok = True; break
                    except Exception:
                        continue
                if not ok:
                    continue
                for _s, _p, o_ in g.triples((None, RDFS.label, None)):
                    words |= tokenize(str(o_))
                for _s, _p, o_ in g.triples((None, RDFS.comment, None)):
                    words |= tokenize(str(o_))
                for s_ in set(g.subjects()):
                    if isinstance(s_, URIRef):
                        words |= tokenize(_localname(str(s_)))
        docs[stem] = " ".join(sorted(words)) if words else stem
    return docs


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------

class EmbeddingItemRecommender(Recommender):
    """Embed ontology vs each catalog item's document; rank by cosine.

    item_docs / item_to_stem override the dataset catalog -> lets the SAME model
    run over ODPReco's FIXED catalog (pass the 73 ODP descriptions + the map).
    """
    uses_dataset = True

    def __init__(self, embspec: str = "st:all-MiniLM-L6-v2",
                 item_docs: dict[str, str] | None = None,
                 item_to_stem: dict[str, str] | None = None,
                 label: str | None = None):
        self.emb = Embedder(embspec)
        self._fixed_docs = item_docs
        self._fixed_map = item_to_stem
        self.name = label or f"embed_item[{embspec}]"
        if item_docs is not None:
            self.uses_dataset = False  # fixed catalog = not learned from data

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        if self._fixed_docs is not None:
            self.items = list(self._fixed_docs.keys())
            self.item_to_stem = dict(self._fixed_map or {})
            docs = [self._fixed_docs[i] for i in self.items]
        else:
            pdocs = build_pattern_docs(train)
            self.items = [s for s in catalog if s in pdocs] or list(catalog)
            self.item_to_stem = {s: s for s in self.items}
            docs = [pdocs.get(s, s) for s in self.items]
        self.item_mat = self.emb.embed(docs)

    def recommend(self, onto: Ontology, top_n: int = 10):
        q = self.emb.embed([onto.text or onto.name])
        sims = _cos(q, self.item_mat)[0]
        order = np.argsort(-sims)
        out, seen = [], set()
        for idx in order:
            stem = self.item_to_stem.get(self.items[idx])
            if stem and stem not in seen:
                seen.add(stem)
                out.append(stem)
            if len(out) >= top_n:
                break
        return out


class EmbeddingOntologyKNN(Recommender):
    """CF with embedding-cosine ontology similarity (vs the Jaccard baseline)."""
    name = "embed_ontology_knn_cf"
    uses_dataset = True

    def __init__(self, embspec: str = "st:all-MiniLM-L6-v2", k: int = 15):
        self.emb = Embedder(embspec)
        self.k = k

    def fit(self, train, catalog):
        self.catalog = list(catalog)
        self.train = train
        self.train_mat = self.emb.embed([o.text or o.name for o in train])

    def recommend(self, onto: Ontology, top_n: int = 10):
        q = self.emb.embed([onto.text or onto.name])
        sims = _cos(q, self.train_mat)[0]
        order = np.argsort(-sims)[: self.k]
        score: dict[str, float] = {}
        for idx in order:
            s = float(sims[idx])
            if s <= 0:
                continue
            for stem in self.train[idx].gt_stems:
                score[stem] = score.get(stem, 0.0) + s
        ranked = sorted(score, key=lambda x: score[x], reverse=True)
        return ranked[:top_n]


class Doc2VecRecommender(Recommender):
    """ODPReco-style content similarity done correctly (gensim Doc2Vec).

    Trains one Doc2Vec model on the multi-line corpus of TRAIN ontology docs +
    pattern docs, infers a vector for the test ontology, ranks pattern docs by
    cosine. Unlike ODPReco (1-line corpora, model thrown away each run, hardcoded
    73), this trains a real model over the whole dataset.
    """
    name = "doc2vec"
    uses_dataset = True

    def __init__(self, vector_size: int = 100, epochs: int = 40, seed: int = 1):
        self.vector_size = vector_size
        self.epochs = epochs
        self.seed = seed

    def fit(self, train, catalog):
        # gensim<4.3.3 imports triu/tril from scipy.linalg, removed in scipy>=1.13.
        import scipy.linalg as _sla
        if not hasattr(_sla, "triu"):
            _sla.triu = np.triu
        if not hasattr(_sla, "tril"):
            _sla.tril = np.tril
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        self.catalog = list(catalog)
        pdocs = build_pattern_docs(train)
        self.stems = [s for s in catalog if s in pdocs] or list(catalog)
        docs = []
        for o in train:
            docs.append(TaggedDocument(list(tokenize(o.text or o.name)),
                                       [f"O::{o.name}"]))
        for s in self.stems:
            docs.append(TaggedDocument(list(tokenize(pdocs.get(s, s))),
                                       [f"P::{s}"]))
        self.model = Doc2Vec(docs, vector_size=self.vector_size,
                             min_count=1, epochs=self.epochs, workers=1,
                             seed=self.seed, dm=1, window=5)
        self.pat_vecs = {}
        for s in self.stems:
            tag = f"P::{s}"
            if tag in self.model.dv:
                v = self.model.dv[tag]
                n = np.linalg.norm(v)
                self.pat_vecs[s] = v / n if n else v

    def recommend(self, onto: Ontology, top_n: int = 10):
        toks = list(tokenize(onto.text or onto.name))
        if not toks or not self.pat_vecs:
            return self.stems[:top_n]
        q = self.model.infer_vector(toks, epochs=self.epochs)
        n = np.linalg.norm(q)
        q = q / n if n else q
        scored = sorted(self.pat_vecs.items(),
                        key=lambda kv: float(q @ kv[1]), reverse=True)
        return [s for s, _ in scored[:top_n]]
