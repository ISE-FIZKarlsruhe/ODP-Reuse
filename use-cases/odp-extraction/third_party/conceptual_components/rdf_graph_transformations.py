# Vendored from stlab-istc-cnr/conceptual-components
# @ e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5
# path: conceptual-components-extraction/project_root/rdf_graph_transformations.py
# blob: f6d30136a7268361569ba19d834b774781900842
#
# Only the functions on the step-1 execution path are copied, verbatim.  The
# entry point `rdf_greedy_modularity_communities_with_rdf_from_folder` calls
# `create_owlthing_nodisjoint_class_distinctproperty_nosubsumed`, which calls
# `create_graph_w_owlthing`,
# `create_relation_and_subsumed_intensional_graph_without_annotations` and
# `create_class_distinctproperty_graph`; those four, and nothing else, are here.
# The SPARQL text is byte-identical to upstream.  See PROVENANCE.md.

from rdflib import *          # noqa: F401,F403  (upstream style)
import os                     # noqa: F401       (upstream import, kept)
from .utils import getID


# this function takes an rdf graph as input and returns an intensional graph
# ignoring subclassof, equivalentclass and disjoint axioms,
# by performing sparql queries
def create_relation_and_subsumed_intensional_graph_without_annotations(graph):
    relation_intensional_results_no_class_construction = graph.query("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX fschema: <https://w3id.org/framester/schema/>

    CONSTRUCT {
    ?class1 fschema:subsumedUnder ?class2 .
    ?equal1 fschema:extensionallyEquivalent ?equal2  .
    ?domain1 ?domain2 ?domain3 .
    ?universal1 ?universal2 ?universal3 .
    ?existential1 ?existential2 ?existential3 .
    ?card1 ?card2 ?card3 .
    }

    WHERE {
    {
    ?class1 rdfs:subClassOf ?class2 .
    FILTER NOT EXISTS {?class2 a owl:Restriction}
    FILTER NOT EXISTS {?class2 owl:unionOf ?u}
    FILTER NOT EXISTS {?class2 owl:intersectionOf ?i}
    FILTER NOT EXISTS {?class2 owl:oneOf ?o}
    FILTER NOT EXISTS {?class2 owl:complementOf ?c}
    }
    UNION
    {
    ?equal1 owl:equivalentClass ?equal2 .
    FILTER NOT EXISTS {?equal2 a owl:Restriction}
    FILTER NOT EXISTS {?equal2 owl:unionOf ?u}
    FILTER NOT EXISTS {?equal2 owl:intersectionOf ?i}
    FILTER NOT EXISTS {?equal2 owl:oneOf ?o}
    FILTER NOT EXISTS {?equal2 owl:complementOf ?c}
    FILTER(!isBlank(?equal2))
    FILTER(!isBlank(?equal1))
    }
    UNION
    {
    ?allequivalent a owl:AllEquivalentClasses ;
    owl:members/rdf:rest*/rdf:first ?equal1 ; owl:members/rdf:rest*/rdf:first ?equal2 .
    FILTER(?equal1 != ?equal2)
    }
    UNION
    {
    ?domain2 rdfs:domain ?domain1 . ?domain2 rdfs:range ?domain3 .
    FILTER NOT EXISTS {?domain1 owl:unionOf|owl:intersectionOf|owl:complementOf ?z}
    FILTER NOT EXISTS {?domain3 owl:unionOf|owl:intersectionOf|owl:complementOf ?z}
    FILTER(!isBlank(?domain1))
    FILTER(!isBlank(?domain2))
    FILTER(!isBlank(?domain3))
    }
    UNION
    {
    ?universal1 rdfs:subClassOf|owl:equivalentClass ?z .
    ?z owl:onProperty ?universal2 ; owl:allValuesFrom ?universal3 .
    FILTER NOT EXISTS {?universal2 a owl:Restriction}
    FILTER NOT EXISTS {?universal3 a owl:Restriction}
    FILTER NOT EXISTS {?universal3 owl:unionOf ?u}
    FILTER NOT EXISTS {?universal3 owl:intersectionOf ?i}
    FILTER NOT EXISTS {?universal3 owl:complementOf ?c}
    FILTER(!isBlank(?universal2))
    FILTER(!isBlank(?universal3))
    }
    UNION
    {
    ?existential1 rdfs:subClassOf|owl:equivalentClass ?z .
    ?z owl:onProperty ?existential2 ; owl:someValuesFrom ?existential3 .
    FILTER NOT EXISTS {?existential2 a owl:Restriction} .
    FILTER NOT EXISTS {?existential3 a owl:Restriction} .
    FILTER NOT EXISTS {?existential3 owl:unionOf ?u}
    FILTER NOT EXISTS {?existential3 owl:intersectionOf ?i}
    FILTER NOT EXISTS {?existential3 owl:complementOf ?c}
    FILTER(!isBlank(?existential2))
    FILTER(!isBlank(?existential3))
    }
    UNION
    {
    ?card1 rdfs:subClassOf|owl:equivalentClass ?z .
    ?z owl:onProperty ?card2 ; owl:minCardinality|owl:maxCardinality|owl:cardinality ?q .
    ?card2 rdfs:range ?card3 .
    FILTER (?q != 0)
    FILTER(!isBlank(?card2))
    FILTER(!isBlank(?card3))
    }
    }
    """)

    subsumed_intensional_graph = Graph()
    for triple in relation_intensional_results_no_class_construction:
        subsumed_intensional_graph.add(triple)

    relation_intensional_graph = Graph()
    for s, p, o in subsumed_intensional_graph:
        if not p == URIRef("https://w3id.org/framester/schema/extensionallyEquivalent") and not p == URIRef("https://w3id.org/framester/schema/subsumedUnder"):
            relation_intensional_graph.add((s, p, o))

    return relation_intensional_graph, subsumed_intensional_graph


# this function takes an rdf graph as input and returns a property graph
# i.e. all rdf properties in input graph become nodes, along with classes
# e.g. C1 -p1-> C2
# becomes
# C1 -edge-> p1
# p1 -edge-> C2
# but properties are made distinct by concatenating the original triple they were property of
def create_class_distinctproperty_graph(graph):
    pred_dict = {}

    edge = URIRef("https://w3id.org/framester/schema/edge")
    # TODO transform http://edge.example/ into constant
    prefix = "http://edge.example/"
    property_graph = Graph()
    for s, p, o in graph.triples((None, None, None)):
        pred = URIRef(prefix + getID(s) + "--" + getID(p) + "--" + getID(o))
        property_graph.add((s, edge, pred))
        property_graph.add((pred, edge, o))
        pred_dict[str(pred)] = [str(s), str(p), str(o)]
    return property_graph, pred_dict, graph


# this function takes an rdf graph as input and returns a graph with additional
# rdfs:domain and rdfs:range restrictions
# with owl:Thing as object
# when a property has no domain and/or range
def create_graph_w_owlthing(graph):
    owlthing_graph_results = graph.query("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX fschema: <https://w3id.org/framester/schema/>

    CONSTRUCT {
    ?property1 rdfs:domain owl:Thing .
    ?property2 rdfs:range owl:Thing .
    }

    WHERE {
    {
    ?property1 a owl:ObjectProperty
    FILTER NOT EXISTS {?property1 rdfs:domain ?domain}
    }
    UNION
    {
    ?property2 a owl:ObjectProperty
    FILTER NOT EXISTS {?property2 rdfs:range ?range}
    }
    }
    """)

    owlthing_graph = Graph()
    for triple in owlthing_graph_results:
        owlthing_graph.add(triple)

    return owlthing_graph


### combinations of queries
def create_owlthing_nodisjoint_class_distinctproperty_nosubsumed(graph):
    owl_thing_graph = graph + create_graph_w_owlthing(graph)
    nosubsumed_intensional_graph = create_relation_and_subsumed_intensional_graph_without_annotations(owl_thing_graph)[0]
    class_distinctproperty_nosubsumed_intensional_graph = create_class_distinctproperty_graph(nosubsumed_intensional_graph)
    subsumed_intensional_graph = create_relation_and_subsumed_intensional_graph_without_annotations(owl_thing_graph)[1]

    return class_distinctproperty_nosubsumed_intensional_graph, subsumed_intensional_graph
