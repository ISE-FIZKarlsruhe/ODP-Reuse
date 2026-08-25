"""Asprino et al. (K-CAP 2021) conceptual-components extraction, step 1.

This is a thin adapter over the authors' own code, vendored in
``third_party/conceptual_components/`` (see its ``PROVENANCE.md`` for the
upstream commit, blob hashes and the exact patches needed to run 2021 code under
Python 3.11 / rdflib 7 / networkx 3). No SPARQL query, no parameter and no
algorithm here is a reimplementation: the graph construction, the
Clauset--Newman--Moore greedy modularity maximisation and its recursive
refinement criterion are upstream's.

WHAT STEP 1 DOES
----------------
1. Add ``rdfs:domain owl:Thing`` / ``rdfs:range owl:Thing`` for object properties
   that declare neither, so every property participates in the graph.
2. Build the *intensional graph* by SPARQL CONSTRUCT: domain/range pairs,
   universal and existential restrictions, cardinality restrictions -- and then
   drop the ``subsumedUnder`` / ``extensionallyEquivalent`` edges, i.e.
   subsumption and equivalence do NOT enter the graph that is clustered.
3. Reify every remaining triple ``s p o`` into an *edge node*
   ``http://edge.example/s--p--o`` with ``s -- edge --> node -- edge --> o``, so
   properties become nodes alongside classes.
4. Convert to an undirected networkx graph and run greedy modularity
   maximisation, recursively re-running it on any community whose density is
   below the average density of its siblings.
5. Map each community's nodes back to original IRIs, expanding edge nodes to
   their ``(s, p, o)``.

WHAT IS AND IS NOT EVALUATED
----------------------------
The method's *published* output is a corpus-level catalogue of conceptual
components, obtained by clustering these communities across ontologies (upstream
steps 2--7: virtual documents, UKB word-sense disambiguation, Framester frame
detection, k-means). That catalogue is not the quantity ODP-Reuse scores.
ODP-Reuse scores a per-ontology grouping of entities, and that is exactly what
step 1 emits, with no dependency on steps 2--7. Everything reported from this
module is therefore labelled "step 1" and never "Asprino et al." unqualified.

ENTITY UNIVERSE
---------------
A community can contain IRIs that are not entities under this benchmark's WIDE
definition (``owl:Thing``, predicates such as ``rdf:type``, annotation
vocabulary). ``extract`` restricts every community to the ``entities`` list the
caller passes -- the same universe the embedding baselines cluster over and the
same universe the ground truth lives in -- so precision is not penalised for a
disagreement about what an entity is. Set ``restrict_to_entities=False`` to score
the raw communities instead; ``run_method_eval.py`` reports both.
"""

from __future__ import annotations

import sys
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
if str(_USECASE) not in sys.path:
    sys.path.insert(0, str(_USECASE))

from third_party.conceptual_components.compute_greedy_modularity_communities import (  # noqa: E402
    print_original_uris_from_communities,
    print_recursive_greedy_modularity_communities,
)
from third_party.conceptual_components.rdf_graph_transformations import (  # noqa: E402
    create_owlthing_nodisjoint_class_distinctproperty_nosubsumed,
)
from third_party.conceptual_components.rdflib_to_networkx_graph import (  # noqa: E402
    from_rdflib_to_networkx_graph,
)

METHOD_NAME = "asprino_cc"


def communities(graph) -> list[set[str]]:
    """Run upstream step 1 on one rdflib graph; return communities of IRIs.

    Mirrors ``rdf_greedy_modularity_communities_with_rdf_from_folder``'s
    ``recursive="True"`` branch, which is the setting the upstream README
    prescribes and the one used for the published results.
    """
    built = create_owlthing_nodisjoint_class_distinctproperty_nosubsumed(graph)
    property_graph = built[0][0]
    predicates_originaluris_dict = built[0][1]

    if len(property_graph) == 0:
        return []

    networkx_graph = from_rdflib_to_networkx_graph(property_graph)
    comms = print_recursive_greedy_modularity_communities(networkx_graph)
    return print_original_uris_from_communities(comms, predicates_originaluris_dict)


def extract(graph, entities: list[str], labels: dict[str, str] | None = None,
            restrict_to_entities: bool = True) -> dict[int, set[str]]:
    """Predicted clusters for one ontology, in the runner's cluster format.

    ``labels`` is accepted and ignored; it exists so every method in this use
    case has one call signature.
    """
    universe = set(entities)
    out: dict[int, set[str]] = {}
    for i, comm in enumerate(communities(graph)):
        members = (comm & universe) if restrict_to_entities else set(comm)
        if members:
            out[i] = members
    return out
