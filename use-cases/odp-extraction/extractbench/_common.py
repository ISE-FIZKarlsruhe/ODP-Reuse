# VENDORED from PatternMiner rq_0_cache/embedders/_common.py -- do not edit.
# Copied verbatim so both papers use byte-identical embedders.
"""Shared helpers for embedder adapters."""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from rdflib import BNode, Graph, OWL, RDF, RDFS, URIRef


def local_name(iri: str) -> str:
    s = str(iri)
    for sep in ("#", "/"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s


def camel_to_words(name: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ")


def fallback_label(iri: str) -> str:
    """Use labels-dict if available, else CamelCase-split local name."""
    return camel_to_words(local_name(iri))


def build_adj(g: Graph, entity_set: set[str]) -> dict[str, list[str]]:
    """Adjacency list over IRI-to-IRI triples among entities."""
    adj: dict[str, list[str]] = defaultdict(list)
    for s, _, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            hs, os_ = str(s), str(o)
            if hs in entity_set and os_ in entity_set:
                adj[hs].append(os_)
    return adj


def collect_triples(g: Graph, entity_set: set[str]) -> list[tuple[str, str, str]]:
    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            hs, ps_, os_ = str(s), str(p), str(o)
            if hs in entity_set and os_ in entity_set:
                triples.append((hs, ps_, os_))
    return triples


def hierarchy_edges(g: Graph, entity_set: set[str]) -> list[tuple[str, str]]:
    """subClassOf + domain + range + equivalentClass as (parent, child) pairs."""
    edges = []
    for pred in [RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.equivalentClass]:
        for s, _, o in g.triples((None, pred, None)):
            if isinstance(s, URIRef) and isinstance(o, URIRef):
                a, b = str(s), str(o)
                if a in entity_set and b in entity_set:
                    edges.append((a, b))
    return edges


def scipy_triu_patch() -> None:
    """gensim may import scipy.linalg.triu (removed in scipy 1.12+); patch it."""
    try:
        import scipy.linalg
        if not hasattr(scipy.linalg, "triu"):
            scipy.linalg.triu = np.triu  # type: ignore[attr-defined]
    except ImportError:
        pass
