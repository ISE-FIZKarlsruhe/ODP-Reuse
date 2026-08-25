# Vendored from stlab-istc-cnr/conceptual-components
# @ e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5
# path: conceptual-components-extraction/project_root/rdflib_to_networkx_graph.py
#
# PATCH 1 (see PROVENANCE.md): upstream's `import rdfextras` is removed.
# rdfextras 0.4 is a Python-2-era package that no longer installs, and nothing
# in this module references it -- it is an unused import.
#
# Everything else is verbatim; only the functions on the step-1 execution path
# are copied.

from rdflib import *                              # noqa: F401,F403
from rdflib.extras.external_graph_libs import *   # noqa: F401,F403
import networkx as nx
from .rdf_graph_transformations import *          # noqa: F401,F403


# this function takes as input an rdflib graph and convert it to a networkx
# digraph
# where node labels are converted to integers
# the networkx graph stores old node labels in an attribute named "uri"
def from_rdflib_to_networkx_digraph(rdf_graph):
    networkx_graph = rdflib_to_networkx_digraph(rdf_graph)
    networkx_graph_int = nx.convert_node_labels_to_integers(networkx_graph, first_label=0, ordering='sorted',
                                                            label_attribute='uri')
    return networkx_graph_int


# this function takes as input an rdflib graph and convert it to a networkx
# graph
# where node labels are converted to integers
# the networkx graph stores old node labels in an attribute named "uri"
def from_rdflib_to_networkx_graph(rdf_graph):
    networkx_graph = rdflib_to_networkx_graph(rdf_graph)
    networkx_graph_int = nx.convert_node_labels_to_integers(networkx_graph, first_label=0, ordering='sorted',
                                                            label_attribute='uri')
    return networkx_graph_int
