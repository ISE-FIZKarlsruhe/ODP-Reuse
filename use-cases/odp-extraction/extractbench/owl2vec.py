# VENDORED from PatternMiner rq_0_cache/embedders/owl2vec.py -- do not edit.
# Copied verbatim so both papers use byte-identical embedders.
"""OWL2Vec* embedder (simplified).

Reference: Chen et al., Artificial Intelligence 2021.

This adapter trains Word2Vec skip-gram on random walks over the ontology's
IRI-to-IRI adjacency (structural walks). It does not reproduce the full
OWL2Vec* pipeline (which also uses lexical walks over labels and a mixed
vocabulary), but matches the structural component used throughout alt0 /
alt50 / etc. in this codebase.
"""
from __future__ import annotations

import numpy as np
from rdflib import Graph

from ._common import build_adj, scipy_triu_patch

METHOD_NAME = "owl2vec"
NATIVE_DIM = 200


def _random_walks(entities, adj, n_walks=100, depth=6, seed=42):
    rng = np.random.default_rng(seed)
    walks = []
    for iri in entities:
        for _ in range(n_walks):
            walk = [iri]
            cur = iri
            for _ in range(depth):
                nbrs = adj.get(cur, [])
                if not nbrs:
                    break
                cur = nbrs[rng.integers(len(nbrs))]
                walk.append(cur)
            walks.append(walk)
    return walks


def embed(graph: Graph, entities: list[str], labels: dict[str, str]) -> np.ndarray:
    scipy_triu_patch()
    from gensim.models import Word2Vec

    entity_set = set(entities)
    adj = build_adj(graph, entity_set)
    walks = _random_walks(entities, adj, n_walks=100, depth=6)
    if not walks:
        return np.zeros((len(entities), NATIVE_DIM), dtype=np.float32)

    model = Word2Vec(
        sentences=walks,
        vector_size=NATIVE_DIM,
        window=5,
        min_count=1,
        sg=1,  # skip-gram
        epochs=50,
        negative=25,
        seed=42,
        workers=1,
    )
    emb = np.zeros((len(entities), NATIVE_DIM), dtype=np.float32)
    for i, iri in enumerate(entities):
        if iri in model.wv:
            emb[i] = model.wv[iri]
    return emb
