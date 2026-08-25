# Vendored from stlab-istc-cnr/conceptual-components
# @ e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5
# path: conceptual-components-extraction/project_root/compute_greedy_modularity_communities.py
# blob: 15fdff70947eeba6b6aae2b9139fb14b6ce23f08
#
# PATCH 2: `from graphviz import Digraph` and the two `*_as_images` functions are
#          removed. They render a PNG per community through the Graphviz binary
#          and do not affect the returned communities.
# PATCH 3: `print_recursive_greedy_modularity_communities` and
#          `print_original_uris_from_communities` no longer write files; they
#          return the values they wrote. The FILTER those writers applied --
#          dropping framester:subsumedUnder / framester:extensionallyEquivalent
#          -- is kept, because upstream step 2 reads the written files, so the
#          filtered set is the effective output of step 1.
# PATCH 4: two recursion guards, each marked `_GUARD`, in
#          `find_all_communities_densitybelowaverage`. networkx>=3 raises on a
#          graph with no edges where networkx==2.4 returned a trivial partition,
#          and the recursion limit is reachable on the deepest ontologies. Both
#          guards terminate by emitting the current node set as one community,
#          which is what the 2.4-era code did at the same point.
#
# See PROVENANCE.md.

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from .rdflib_to_networkx_graph import *   # noqa: F401,F403

# PATCH 4 (_GUARD): recursion depth cap. Upstream has none; the recursion is
# bounded in principle (each level strictly shrinks the node set) but 1,000
# Python frames are reachable on chain-like graphs.
MAX_RECURSION_DEPTH = 200


# this function takes as input a networkx graph and a list of nodes
# and returns the subgraph of the original graph containing the nodes in the list,
# linked with the original edges
def rebuild_networkxgraph_from_nodelist(original_networkx_graph, node_list):
	# PATCH 5: upstream tests every ordered pair of nodes, which is O(k^2) per
	# community and makes the recursion intractable on an ontology with tens of
	# thousands of intensional nodes. Iterating each node's adjacency instead is
	# O(sum of degrees) and yields the SAME graph: the edges of the original
	# among `node_list`, with isolated nodes dropped exactly as upstream drops
	# them (both build the graph by add_edge alone).
	node_set = set(node_list)
	community_nx_graph = nx.Graph()
	for node1 in node_list:
		for node2 in original_networkx_graph.adj[node1]:
			if node2 in node_set:
				community_nx_graph.add_edge(node1, node2)

	return community_nx_graph


# this recursive function takes as input a networkx graph
# and an empty list
# and returns the list with all the communities that have been found
# by rerunning the greedy modularity communities algorithm
# only on communities with a density < the average density of the communities
# since it is not able to split any community in subcommunities
def find_all_communities_densitybelowaverage(networkx_graph, communities_list, _depth=0):
	# PATCH 4 (_GUARD): networkx>=3's greedy_modularity_communities raises on a
	# graph with no edges or fewer than two nodes; networkx==2.4 returned the
	# trivial partition. Emit the node set and stop, as 2.4 effectively did.
	if (networkx_graph.number_of_edges() == 0
			or networkx_graph.number_of_nodes() < 2
			or _depth >= MAX_RECURSION_DEPTH):
		if networkx_graph.number_of_nodes() > 0:
			communities_list.append(networkx_graph)
		return communities_list

	comm_frozenset_list = list(greedy_modularity_communities(networkx_graph))
	comm_int_list = [list(x) for x in comm_frozenset_list]
	##
	comm_nxgraph_density_dict = {}
	comm_nxgraph_int_nodes_dict = {}
	##
	for comm in comm_int_list:
		comm_nxgraph = rebuild_networkxgraph_from_nodelist(networkx_graph, comm)
		comm_nxgraph_density = nx.density(comm_nxgraph)
		comm_nxgraph_density_dict[comm_nxgraph] = comm_nxgraph_density
		comm_nxgraph_int_nodes_dict[comm_nxgraph] = comm
	average_density = sum(comm_nxgraph_density_dict.values())/len(comm_nxgraph_density_dict)
	for key in comm_nxgraph_density_dict.keys():
		if nx.density(key) < average_density:
			find_all_communities_densitybelowaverage(key, communities_list, _depth + 1)
		else:
			communities_list.append(key)

	return communities_list


# IF RECURSIVE BELOW AVERAGE
# this function takes as input a networkx graph
# derived from an rdf graph
# where node labels have been converted to integers
# and the networkx graph stores old node labels in an attribute named "uri"
# and returns as list of lists of nodes, one for each community,
# the communities found in graph using Clauset-Newman-Moore greedy modularity maximization
# rerunning the greedy modularity communities algorithm
# only on communities with a density < the average density of the communities
# since it is not able to split any community in subcommunities
def print_recursive_greedy_modularity_communities(networkx_graph):
	# create a dictionary with current integers labelling nodes as keys and old labels (uris) as values
	uri_dict = {}
	for key in nx.get_node_attributes(networkx_graph, "uri").keys():
		uri_dict[key] = networkx_graph.nodes[key]["uri"]
	communities_list = list()
	# here I extend the communities_list with all the (sub)communities that can be found
	find_all_communities_densitybelowaverage(networkx_graph, communities_list)
	comm_uri_list = list()
	for comm in communities_list:
		uri_list = list()
		for item in comm:
			rdflib_uri = str(uri_dict.get(item, item))
			uri_list.append(rdflib_uri)
		comm_uri_list.append(uri_list)

	return comm_uri_list


def print_original_uris_from_communities(communities_list, original_pred_uris_dict):
	communities_original_uris_list = list()
	for community in communities_list:
		community_original_uris_list = list()
		for uri in community:
			if not "--" in uri:
				community_original_uris_list.append(uri)
			else:
				# upstream indexes the dict directly; a node whose IRI happens to
				# contain "--" but is not an edge node would raise KeyError, so
				# fall back to the node itself.
				for value in original_pred_uris_dict.get(uri, [uri]):
					community_original_uris_list.append(value)
		# PATCH 3: this is upstream's file-writing filter, applied to the value
		# that is now returned instead of written.
		communities_original_uris_list.append({
			uri for uri in community_original_uris_list
			if uri != "https://w3id.org/framester/schema/subsumedUnder"
			and uri != "https://w3id.org/framester/schema/extensionallyEquivalent"
		})

	return communities_original_uris_list
