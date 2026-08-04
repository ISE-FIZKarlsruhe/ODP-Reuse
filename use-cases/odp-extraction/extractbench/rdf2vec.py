# VENDORED from PatternMiner rq_0_cache/embedders/rdf2vec.py -- do not edit.
# Copied verbatim so both papers use byte-identical embedders.
"""RDF2Vec embedder (simplified CBOW over raw RDF walks).

Reference: Ristoski et al., Semantic Web 2019.

Uses biased random walks + Word2Vec CBOW. Walks traverse raw RDF triples
(including property IRIs, not just entity-to-entity).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from rdflib import Graph, URIRef

from ._common import scipy_triu_patch

METHOD_NAME = "rdf2vec"
NATIVE_DIM = 100


def _walks_rdf(entities, g: Graph, entity_set, n_walks=100, depth=4, seed=42):
    """Walks that alternate entity -> predicate -> entity."""
    # Build adjacency including predicate IRIs
    adj = defaultdict(list)
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            a, b = str(s), str(o)
            if a in entity_set and b in entity_set:
                adj[a].append((str(p), b))
    rng = np.random.default_rng(seed)
    walks = []
    for iri in entities:
        for _ in range(n_walks):
            walk = [iri]
            cur = iri
            for _ in range(depth):
                nbrs = adj.get(cur)
                if not nbrs:
                    break
                p_iri, next_e = nbrs[rng.integers(len(nbrs))]
                walk.append(p_iri)
                walk.append(next_e)
                cur = next_e
            walks.append(walk)
    return walks


def embed(graph: Graph, entities: list[str], labels: dict[str, str]) -> np.ndarray:
    scipy_triu_patch()
    from gensim.models import Word2Vec

    entity_set = set(entities)
    walks = _walks_rdf(entities, graph, entity_set, n_walks=100, depth=4)
    if not walks:
        return np.zeros((len(entities), NATIVE_DIM), dtype=np.float32)

    model = Word2Vec(
        sentences=walks,
        vector_size=NATIVE_DIM,
        window=5,
        min_count=1,
        sg=0,  # CBOW (classical RDF2Vec default)
        epochs=30,
        negative=25,
        seed=42,
        workers=1,
    )
    emb = np.zeros((len(entities), NATIVE_DIM), dtype=np.float32)
    for i, iri in enumerate(entities):
        if iri in model.wv:
            emb[i] = model.wv[iri]
    return emb
