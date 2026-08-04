"""
Entity collection and annotation preprocessing.

The entity-set definitions here are deliberately byte-compatible with
PatternMiner's ``rq_0_cache/build_cache.py::_collect_entities`` so that ontology
and entity counts reconcile across the two papers. Two sets are defined:

NARROW
    IRIs that are *declared* as owl:Class / rdfs:Class / owl:ObjectProperty /
    owl:DatatypeProperty, plus any subject of rdfs:subClassOf.

WIDE   (the authoritative set; PatternMiner's clean run used ODP_WIDE_ENTITIES=1)
    NARROW plus IRIs that occur only in a class/property/individual *position*:
    subClassOf objects, equivalentClass/disjointWith on either side, restriction
    fillers (someValuesFrom / allValuesFrom / onClass / onProperty), rdfs:domain
    and rdfs:range, intersectionOf / unionOf members, and owl:NamedIndividual.
    owl:Thing and owl:Nothing are removed.

ANNOTATION PREPROCESSING
    PatternMiner filters annotation vocabularies on the *pattern* side
    (``patternminer/patterns/canonical.py``) but not on the *host ontology*
    side, so dcterms/skos/dc edges become random-walk edges and KGE training
    triples for the host but not for the pattern. That asymmetry biases every
    host-vs-pattern comparison. ``strip_annotations`` removes those triples so
    both sides are treated identically. It is applied before entity collection
    and before any graph is handed to an embedder.
"""

from __future__ import annotations

from rdflib import BNode, Graph, Literal, OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection

# Namespaces whose triples carry metadata, not modelling content. Mirrors
# PatternMiner's ANNOTATION_NAMESPACES (SKOS / DC / DCTERMS / DCAM / CPANN) and
# adds the vocabularies actually observed in this corpus.
ANNOTATION_NAMESPACES: tuple[str, ...] = (
    "http://www.w3.org/2004/02/skos/core#",
    "http://purl.org/dc/elements/1.1/",
    "http://purl.org/dc/terms/",
    "http://purl.org/dc/dcam/",
    "http://www.ontologydesignpatterns.org/schemas/cpannotationschema.owl#",
    "http://purl.org/vocab/vann/",
    "http://xmlns.com/foaf/0.1/",
    "http://creativecommons.org/ns#",
    "http://purl.org/pav/",
    "http://www.w3.org/ns/prov#",
    "http://schema.org/",
    "http://rdfs.org/ns/void#",
)

# Individual annotation predicates outside those namespaces.
ANNOTATION_PREDICATES: frozenset[URIRef] = frozenset({
    RDFS.comment, RDFS.label, RDFS.seeAlso, RDFS.isDefinedBy,
    OWL.versionInfo, OWL.priorVersion, OWL.backwardCompatibleWith,
    OWL.incompatibleWith, OWL.deprecated,
    URIRef("http://www.w3.org/2002/07/owl#annotatedSource"),
    URIRef("http://www.w3.org/2002/07/owl#annotatedProperty"),
    URIRef("http://www.w3.org/2002/07/owl#annotatedTarget"),
})

DECLARED_TYPES = (OWL.Class, RDFS.Class, OWL.ObjectProperty, OWL.DatatypeProperty)


def is_annotation_predicate(p: URIRef) -> bool:
    if p in ANNOTATION_PREDICATES:
        return True
    ps = str(p)
    return any(ps.startswith(ns) for ns in ANNOTATION_NAMESPACES)


def strip_annotations(g: Graph) -> tuple[Graph, int]:
    """Return (filtered copy of g, number of triples removed).

    Removes any triple whose predicate is an annotation predicate, and any
    triple whose object is a Literal (literals cannot be entities and are not
    used by the structural embedders). owl:Ontology header triples are dropped
    too: they describe the file, not the modelling.
    """
    out = Graph()
    for prefix, ns in g.namespaces():
        out.bind(prefix, ns)

    removed = 0
    ontology_nodes = {s for s in g.subjects(RDF.type, OWL.Ontology)}
    for s, p, o in g:
        if is_annotation_predicate(p) or isinstance(o, Literal) or s in ontology_nodes:
            removed += 1
            continue
        out.add((s, p, o))
    return out, removed


def collect_entities(g: Graph, wide: bool = True) -> list[str]:
    """Entity IRIs of ``g``. See module docstring for NARROW vs WIDE."""
    ents: set[str] = set()

    for t in DECLARED_TYPES:
        for s, _, _ in g.triples((None, RDF.type, t)):
            if isinstance(s, URIRef):
                ents.add(str(s))
    for s, _, _ in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef):
            ents.add(str(s))

    if not wide:
        return sorted(ents)

    for _s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(o, URIRef):
            ents.add(str(o))
    for p in (OWL.equivalentClass, OWL.disjointWith):
        for s, o in g.subject_objects(p):
            if isinstance(s, URIRef):
                ents.add(str(s))
            if isinstance(o, URIRef):
                ents.add(str(o))
    for p in (OWL.someValuesFrom, OWL.allValuesFrom, OWL.onClass, OWL.onProperty,
              RDFS.domain, RDFS.range):
        for o in g.objects(None, p):
            if isinstance(o, URIRef):
                ents.add(str(o))
    for p in (OWL.intersectionOf, OWL.unionOf):
        for lst in g.objects(None, p):
            try:
                for member in Collection(g, lst):
                    if isinstance(member, URIRef):
                        ents.add(str(member))
            except Exception:  # noqa: BLE001 - malformed collections are common
                pass
    for s, _, _ in g.triples((None, RDF.type, OWL.NamedIndividual)):
        if isinstance(s, URIRef):
            ents.add(str(s))

    ents.discard(str(OWL.Thing))
    ents.discard(str(OWL.Nothing))
    return sorted(ents)


def declared_ontology_iris(g: Graph) -> list[str]:
    """IRIs declared as owl:Ontology (used by the self-match test)."""
    return sorted(str(s) for s in g.subjects(RDF.type, OWL.Ontology)
                  if isinstance(s, URIRef))


_RDF_SUFFIXES = (".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt", ".trig", ".jsonld")


def parse_graph(path) -> tuple[Graph | None, str]:
    """Parse an RDF file, trying formats in turn.

    Returns ``(graph, "")`` on success, or ``(None, error)`` on failure where
    ``error`` is ``"ExceptionType: message"`` for the format that got furthest.

    The exception MESSAGE is preserved, not just the type. A bare
    ``"BadSyntax"`` tells a user nothing actionable about a file that failed to
    parse; the message names the line and the offending token. Callers that want
    only the type can split on the first ``": "``.
    """
    from pathlib import Path
    p = Path(path)
    suffix = p.suffix.lower()
    order = {
        ".ttl": ("turtle", "xml", "n3"),
        ".n3": ("n3", "turtle", "xml"),
        ".nt": ("nt", "turtle"),
        ".trig": ("trig", "turtle"),
        ".jsonld": ("json-ld",),
    }.get(suffix, ("xml", "turtle", "n3", "nt"))

    # Report the error from the format matching the file's own extension -- that
    # is the informative one. Later fallback formats fail for uninteresting
    # reasons ("not well-formed XML" on a Turtle file).
    first_error = ""
    for i, fmt in enumerate(order):
        g = Graph()
        try:
            g.parse(p.as_posix(), format=fmt)
            return g, ""
        except Exception as exc:  # noqa: BLE001
            msg = " ".join(str(exc).split())[:300]
            err = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
            if i == 0:
                first_error = err
    return None, first_error or "ParseError: no format succeeded"
