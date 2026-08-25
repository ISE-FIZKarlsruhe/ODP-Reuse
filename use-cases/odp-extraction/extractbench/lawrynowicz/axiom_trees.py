"""OWL axioms as labelled trees, following Lawrynowicz et al. (SWJ 2018) §4.2.

    Agnieszka Lawrynowicz, Jedrzej Potoniec, Michal Robaczyk, Tania Tudorache.
    "Discovery of emerging design patterns in ontologies using tree mining."
    Semantic Web 9(4):517-544, 2018. doi:10.3233/SW-170280

The paper's encoding, restated:

* Only ``C SubClassOf D`` and ``C EquivalentTo D`` axioms are mined, and only
  those with a **named class on the left-hand side**. Annotation axioms,
  property axioms and individual assertions are out of scope, and
  ``owl:imports`` is ignored ("We used only axioms defined in the ontologies
  themselves").
* Each such axiom becomes **one rooted labelled tree**; each ontology becomes
  one **forest**. The root is the axiom operator; its children are the LHS node
  and the right-hand class expression. Class constructors are internal nodes and
  their arguments are their children.
* Every node is labelled with a **pair (type, name)**. Ten types "allow us to
  preserve the OWL semantics of the names": class constructor, class, datatype,
  object property, datatype property, individual, literal, cardinality, facet,
  and left-hand side. The LHS class carries its own type, so the same IRI in
  subject position and in object position are *different* labels and never match
  each other.
* Entity names are kept verbatim. Nothing is abstracted to a variable before
  mining; variables are inserted afterwards, when a mined subtree is decoded.
* No reasoner is used -- asserted syntax only.

Two deviations from the paper are forced by the surrounding benchmark and are
recorded here rather than buried:

1. The paper reads axioms with the OWL API after normalising serialisations with
   ROBOT. This benchmark parses every ontology with rdflib, because the quality
   gate, the ground truth and the other methods all do, and two parsers would
   mean two entity sets. The RDF encoding of OWL class expressions is
   reconstructed here directly, which covers the same constructors.
2. The paper mines the ontology as published. This benchmark strips annotation
   vocabularies from both sides of every comparison
   (``pipeline/odpqual/entities.py``); annotation axioms are already out of the
   paper's scope, so the two agree on what is mined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdflib import BNode, Literal, OWL, RDF, RDFS, URIRef
from rdflib.collection import Collection

# ---- the ten node types of Sec. 4.2 --------------------------------------
T_CC = "CC"      # class constructor (and the axiom operator itself)
T_C = "C"        # class
T_D = "D"        # datatype
T_OP = "OP"      # object property
T_DP = "DP"      # datatype property
T_I = "I"        # individual
T_LIT = "LIT"    # literal
T_N = "N"        # cardinality (the integer of a cardinality restriction)
T_F = "F"        # facet (of a datatype restriction)
T_LHS = "LHS"    # left-hand side of the axiom (always a named class)

ENTITY_TYPES = frozenset({T_C, T_OP, T_DP, T_I, T_LHS})

SUBCLASS_OF = "SubClassOf"
EQUIVALENT_TO = "EquivalentTo"

# Constructor labels, verbatim from the paper's supplementary pseudocode
# (https://semantic.cs.put.poznan.pl/bioportal-patterns/string-to-manchester.pdf,
# Algorithm 4).
_CARDINALITY_PREDS = {
    OWL.minCardinality: "minCardinality",
    OWL.maxCardinality: "maxCardinality",
    OWL.cardinality: "cardinality",
    OWL.minQualifiedCardinality: "minQualifiedCardinality",
    OWL.maxQualifiedCardinality: "maxQualifiedCardinality",
    OWL.qualifiedCardinality: "qualifiedCardinality",
}

_XSD = "http://www.w3.org/2001/XMLSchema#"

_MAX_DEPTH = 20          # cycle guard; RDF class expressions are trees in OWL
                         # but a malformed file can encode a loop.


@dataclass
class Node:
    type: str
    name: str
    children: list["Node"] = field(default_factory=list)

    @property
    def label(self) -> tuple[str, str]:
        return (self.type, self.name)


@dataclass
class AxiomTree:
    """One SubClassOf / EquivalentTo axiom as a tree."""
    root: Node
    lhs: str                     # IRI of the named class on the left
    entities: frozenset[str]     # every entity IRI occurring in the tree

    def nodes(self) -> list[Node]:
        out: list[Node] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out


# ----------------------------------------------------------------------

class _Encoder:
    def __init__(self, g):
        self.g = g
        self.obj_props = {str(s) for s in g.subjects(RDF.type, OWL.ObjectProperty)}
        self.data_props = {str(s) for s in g.subjects(RDF.type, OWL.DatatypeProperty)}
        self.datatypes = {str(s) for s in g.subjects(RDF.type, RDFS.Datatype)}
        self.individuals = {str(s) for s in g.subjects(RDF.type, OWL.NamedIndividual)}

    # -- leaf typing -----------------------------------------------------
    def _property_node(self, p) -> Node:
        ps = str(p)
        if ps in self.data_props:
            return Node(T_DP, ps)
        return Node(T_OP, ps)          # the paper's tie-break favours object
                                       # properties when the type is unknown

    def _class_node(self, c) -> Node:
        cs = str(c)
        if cs in self.datatypes or cs.startswith(_XSD):
            return Node(T_D, cs)
        if cs in self.individuals:
            return Node(T_I, cs)
        return Node(T_C, cs)

    def _literal_node(self, lit) -> Node:
        return Node(T_LIT, str(lit))

    # -- class expressions ----------------------------------------------
    def expr(self, x, depth: int = 0) -> Node | None:
        if depth > _MAX_DEPTH:
            return None
        if isinstance(x, Literal):
            return self._literal_node(x)
        if isinstance(x, URIRef):
            return self._class_node(x)
        if not isinstance(x, BNode):
            return None

        # owl:Restriction
        prop = self.g.value(x, OWL.onProperty)
        if prop is not None:
            return self._restriction(x, prop, depth)

        for pred, ctor in ((OWL.intersectionOf, "intersectionOf"),
                           (OWL.unionOf, "unionOf")):
            lst = self.g.value(x, pred)
            if lst is not None:
                kids = [k for k in (self.expr(m, depth + 1)
                                    for m in self._members(lst)) if k is not None]
                return Node(T_CC, ctor, kids) if kids else None

        lst = self.g.value(x, OWL.oneOf)
        if lst is not None:
            kids = [Node(T_I, str(m)) if isinstance(m, URIRef)
                    else self._literal_node(m) for m in self._members(lst)]
            return Node(T_CC, "oneOf", kids) if kids else None

        comp = self.g.value(x, OWL.complementOf)
        if comp is not None:
            k = self.expr(comp, depth + 1)
            return Node(T_CC, "complementOf", [k] if k else [])

        if self.g.value(x, OWL.onDatatype) is not None:
            kids: list[Node] = []
            dt = self.g.value(x, OWL.onDatatype)
            if isinstance(dt, URIRef):
                kids.append(Node(T_D, str(dt)))
            wr = self.g.value(x, OWL.withRestrictions)
            if wr is not None:
                for facet_bn in self._members(wr):
                    for fp, fv in self.g.predicate_objects(facet_bn):
                        kids.append(Node(T_F, str(fp)))
                        kids.append(self._literal_node(fv))
            return Node(T_CC, "DatatypeRestriction", kids)

        return None

    def _restriction(self, x, prop, depth: int) -> Node | None:
        pnode = self._property_node(prop)

        for pred, ctor in ((OWL.someValuesFrom, "someValuesFrom"),
                           (OWL.allValuesFrom, "allValuesFrom")):
            filler = self.g.value(x, pred)
            if filler is not None:
                f = self.expr(filler, depth + 1)
                return Node(T_CC, ctor, [pnode] + ([f] if f else []))

        hv = self.g.value(x, OWL.hasValue)
        if hv is not None:
            fill = (Node(T_I, str(hv)) if isinstance(hv, URIRef)
                    else self._literal_node(hv))
            return Node(T_CC, "value", [pnode, fill])

        for pred, ctor in _CARDINALITY_PREDS.items():
            n = self.g.value(x, pred)
            if n is None:
                continue
            kids = [Node(T_N, str(n)), pnode]
            qual = self.g.value(x, OWL.onClass) or self.g.value(x, OWL.onDataRange)
            if qual is not None:
                q = self.expr(qual, depth + 1)
                if q is not None:
                    kids.append(q)
            return Node(T_CC, ctor, kids)

        return None

    def _members(self, lst) -> list:
        try:
            return list(Collection(self.g, lst))
        except Exception:  # noqa: BLE001 - malformed collections are common
            return []


def _collect_entities(root: Node) -> frozenset[str]:
    out: set[str] = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type in ENTITY_TYPES:
            out.add(n.name)
        stack.extend(n.children)
    out.discard(str(OWL.Thing))
    out.discard(str(OWL.Nothing))
    return frozenset(out)


def forest(graph) -> list[AxiomTree]:
    """The forest of one ontology: one tree per mined axiom.

    An axiom is mined when it is ``A rdfs:subClassOf X`` or
    ``A owl:equivalentClass X`` with ``A`` a named class (a URIRef, not a class
    expression). Axioms whose right-hand side yields no encodable expression are
    dropped, as they carry no structure to mine.
    """
    enc = _Encoder(graph)
    trees: list[AxiomTree] = []
    for pred, op in ((RDFS.subClassOf, SUBCLASS_OF),
                     (OWL.equivalentClass, EQUIVALENT_TO)):
        for s, o in graph.subject_objects(pred):
            if not isinstance(s, URIRef):
                continue
            rhs = enc.expr(o)
            if rhs is None:
                continue
            root = Node(T_CC, op, [Node(T_LHS, str(s)), rhs])
            trees.append(AxiomTree(root=root, lhs=str(s),
                                   entities=_collect_entities(root)))
    return trees
