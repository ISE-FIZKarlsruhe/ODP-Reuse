"""Turn a predicted cluster into an OWL module, and into something drawable.

Follows ``PatternMiner/rq_11_pattern_induction/src/robot_extract.py``: ROBOT
Syntactic Locality Module Extraction where ROBOT is available, an rdflib
approximation otherwise. Two things are done differently, both forced by scale.

**The ontology is parsed once, not once per cluster.** The reference calls
``extract_via_python`` per cluster and that function parses the source ontology
each time. The largest ontology here yields 5,204 clusters under one method
alone, so ``TermIndex`` parses once and indexes every triple by the terms it
mentions; slicing a cluster is then proportional to the seeds' degree rather
than to the size of the ontology.

**Blank nodes are followed to their closure.** Selecting triples that mention a
seed cuts every OWL axiom that goes through a blank node -- restrictions,
intersections, RDF lists -- in half, and the result is not valid OWL: a
``rdfs:subClassOf [ owl:onProperty p ; owl:someValuesFrom C ]`` arrives as a
subClassOf pointing at a blank node with nothing on it. The closure is followed
so an axiom is either wholly in the module or wholly absent.

ROBOT's STAR extraction does both of these properly, and is what
``backend="robot"`` uses. It costs a JVM start per cluster, which is why it is
not the default at this corpus's scale.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

OWL = "http://www.w3.org/2002/07/owl#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"

# Predicates that say what an entity *is* rather than how it relates to another
# entity. Kept out of figures, which would otherwise be dominated by every
# class pointing at owl:Class.
STRUCTURAL = {
    RDF + "type", RDFS + "subClassOf", RDFS + "subPropertyOf",
    RDFS + "domain", RDFS + "range",
    OWL + "equivalentClass", OWL + "equivalentProperty",
    OWL + "disjointWith", OWL + "inverseOf", OWL + "onProperty",
    OWL + "someValuesFrom", OWL + "allValuesFrom", OWL + "hasValue",
}
# Of those, the ones that ARE a relation between two entities and belong in a
# figure, with the style to draw them in.
DRAWN_STRUCTURAL = {
    RDFS + "subClassOf": "subsumed",
    RDFS + "subPropertyOf": "subsumed",
    OWL + "equivalentClass": "equivalent",
    OWL + "equivalentProperty": "equivalent",
}


class TermIndex:
    """One parsed ontology, indexed so any cluster can be sliced from it."""

    def __init__(self, graph):
        from rdflib import BNode

        self.graph = graph
        self.by_term = defaultdict(list)       # term -> triples mentioning it
        self.bnode_out = defaultdict(list)     # bnode -> its outgoing triples
        for s, p, o in graph:
            self.by_term[s].append((s, p, o))
            if o is not s:
                self.by_term[o].append((s, p, o))
            if isinstance(s, BNode):
                self.bnode_out[s].append((s, p, o))
        self.namespaces = list(graph.namespaces())

    def module(self, seeds):
        """The rdflib Graph of every axiom mentioning a seed, bnodes closed."""
        from rdflib import BNode, Graph, URIRef

        seed_terms = {URIRef(s) for s in seeds}
        out = Graph()
        for pfx, ns in self.namespaces:
            out.bind(pfx, ns, override=False)

        pending: list = []
        for term in seed_terms:
            for t in self.by_term.get(term, ()):
                out.add(t)
                for end in (t[0], t[2]):
                    if isinstance(end, BNode):
                        pending.append(end)

        seen: set = set()
        while pending:
            b = pending.pop()
            if b in seen:
                continue
            seen.add(b)
            for t in self.bnode_out.get(b, ()):
                out.add(t)
                if isinstance(t[2], BNode):
                    pending.append(t[2])
        return out


def drawable(module, seeds) -> tuple[list, list]:
    """``(triples, singles)`` for ``render_cc.render_graph``.

    A relation is drawn when both endpoints are named entities and at least one
    is a seed, so the figure shows the cluster and what it attaches to rather
    than the whole neighbourhood. ``rdf:type`` and the like are dropped, except
    for the four that genuinely relate two entities.
    """
    from rdflib import URIRef

    seed_terms = {URIRef(s) for s in seeds}
    triples = []
    for s, p, o in module:
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        if s not in seed_terms and o not in seed_terms:
            continue
        ps = str(p)
        if ps in STRUCTURAL and ps not in DRAWN_STRUCTURAL:
            continue
        if str(o).startswith(OWL) or str(o).startswith(RDFS):
            continue                       # owl:Class, rdfs:Resource, ...
        triples.append((str(s), ps, str(o), DRAWN_STRUCTURAL.get(ps, "plain")))

    endpoints = {s for s, _, _, _ in triples} | {o for _, _, o, _ in triples}
    singles = [str(t) for t in sorted(seed_terms, key=str)
               if str(t) not in endpoints]
    return triples, singles


# --------------------------------------------------------------------------
# ROBOT backend -- the reference implementation's preferred path.
# --------------------------------------------------------------------------

def robot_command(robot_jar: str | None = None) -> list[str] | None:
    """The command prefix that runs ROBOT, or None if it cannot be run."""
    if robot_jar:
        jar = Path(robot_jar)
        if jar.is_file() and shutil.which("java"):
            return ["java", "-jar", str(jar)]
        return None
    exe = shutil.which("robot")
    return [exe] if exe else None


def extract_via_robot(cmd: list[str], source: Path, seeds, out_path: Path,
                      method: str = "STAR", timeout: int = 300) -> bool:
    """Syntactic Locality Module Extraction. Returns True if a module was written."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        for iri in seeds:
            fh.write(str(iri).strip() + "\n")
        term_file = fh.name
    try:
        proc = subprocess.run(
            cmd + ["extract", "--method", method.upper(),
                   "--input", str(source), "--term-file", term_file,
                   "--output", str(out_path)],
            capture_output=True, text=True, timeout=timeout)
        return (proc.returncode == 0 and out_path.is_file()
                and out_path.stat().st_size > 0)
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        try:
            Path(term_file).unlink()
        except OSError:
            pass
