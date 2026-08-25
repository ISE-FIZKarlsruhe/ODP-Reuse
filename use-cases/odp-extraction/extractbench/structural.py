"""Structural reference points that bracket the extraction task from below.

None of these is a pattern-extraction method. They exist because a score is only
interpretable against what a method that knows nothing about patterns already
achieves, and because two of them isolate specific properties of the corpus:

``all_in_one``
    One cluster containing every entity. Recall is 1 by construction, so its
    $F_1$ is the score obtainable with no discrimination at all. Any method
    claiming to extract patterns must beat it.
``singletons``
    One cluster per entity. Precision is 1 whenever the entity belongs to the
    matched pattern, recall is $1/|g|$. The complementary degenerate point.
``connected_components``
    Connected components of the ontology's own structure, ignoring the RDF,
    RDFS, OWL and XSD vocabularies so that connectivity does not run through
    ``owl:Class``. Purely structural, no parameters, no learning.
``namespace``
    Entities grouped by IRI namespace. On the plain corpus this is not a
    baseline but a *shortcut*: pattern entities keep the pattern's namespace,
    so grouping by namespace recovers pattern instances almost by definition.
    Its score is therefore an upper reference for what namespace recognition
    alone can achieve, and the gap between its plain and IRI-anonymised scores
    measures how much of any method's score could come from the same shortcut.

All four take the same ``(graph, entities, labels)`` signature as the embedding
methods and return the same ``{cluster_id: {entity_iri}}`` mapping.
"""

from __future__ import annotations

import networkx as nx
from rdflib import BNode, Literal, URIRef

# Vocabulary namespaces whose terms are structure, not domain content. Kept out
# of the co-occurrence graph: with them in, every class is adjacent to
# ``owl:Class`` through ``rdf:type`` and the whole ontology collapses into one
# component, which would make connected_components identical to all_in_one.
_VOCAB_NAMESPACES = (
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "http://www.w3.org/2000/01/rdf-schema#",
    "http://www.w3.org/2002/07/owl#",
    "http://www.w3.org/2001/XMLSchema#",
)


def _is_vocab(term) -> bool:
    s = str(term)
    return any(s.startswith(ns) for ns in _VOCAB_NAMESPACES)


def namespace_of(iri: str) -> str:
    """The IRI up to and including its last ``#`` or ``/``."""
    cut = iri.rfind("#")
    if cut < 0:
        cut = iri.rfind("/")
    return iri[:cut + 1] if cut >= 0 else iri


# ----------------------------------------------------------------------

def all_in_one(graph, entities: list[str], labels=None) -> dict[int, set[str]]:
    return {0: set(entities)} if entities else {}


def singletons(graph, entities: list[str], labels=None) -> dict[int, set[str]]:
    return {i: {e} for i, e in enumerate(entities)}


def connected_components(graph, entities: list[str], labels=None
                         ) -> dict[int, set[str]]:
    """Connected components of the domain-term co-occurrence graph.

    Every triple links the domain terms among its subject, predicate and object;
    blank nodes are ordinary nodes, so a class reaches a restriction filler
    through the restriction's blank node. Components are then projected onto
    ``entities``; components containing no entity are dropped.
    """
    universe = set(entities)
    g = nx.Graph()
    for s, p, o in graph:
        terms = [t for t in (s, p, o)
                 if not isinstance(t, Literal)
                 and (isinstance(t, BNode)
                      or (isinstance(t, URIRef) and not _is_vocab(t)))]
        for i, a in enumerate(terms):
            for b in terms[i + 1:]:
                g.add_edge(a, b)
        for t in terms:
            g.add_node(t)

    out: dict[int, set[str]] = {}
    for comp in nx.connected_components(g):
        members = {str(t) for t in comp if isinstance(t, URIRef)} & universe
        if members:
            out[len(out)] = members
    # Entities that never occur in a retained triple are their own components.
    seen = set().union(*out.values()) if out else set()
    for e in universe - seen:
        out[len(out)] = {e}
    return out


def namespace(graph, entities: list[str], labels=None) -> dict[int, set[str]]:
    """Group entities by IRI namespace. See the module docstring: on the plain
    corpus this exploits exactly the namespace shortcut the anonymised twin
    exists to remove."""
    groups: dict[str, set[str]] = {}
    for e in entities:
        groups.setdefault(namespace_of(e), set()).add(e)
    return {i: v for i, v in enumerate(groups.values())}


METHODS = {
    "all_in_one": all_in_one,
    "singletons": singletons,
    "connected_components": connected_components,
    "namespace": namespace,
}
