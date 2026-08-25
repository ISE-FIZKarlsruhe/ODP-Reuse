# Vendored: Asprino, Carriero & Presutti — conceptual-components extraction

Upstream: <https://github.com/stlab-istc-cnr/conceptual-components>
Commit: `e935c3cdc2a35aa09ebd92cadbfe593ef3a218f5` (2023-01-29, *"update of evaluation
CCs based on ArCo CQs"* — the repository HEAD; there are no later commits).

Paper: Luigi Asprino, Valentina Anita Carriero, Valentina Presutti.
*Extraction of Common Conceptual Components from Multiple Ontologies.*
K-CAP 2021. <https://arxiv.org/abs/2106.12831>

Upstream blob hashes of the files this directory derives from:

| upstream path | git blob sha |
|---|---|
| `conceptual-components-extraction/project_root/compute_greedy_modularity_communities.py` | `15fdff70947eeba6b6aae2b9139fb14b6ce23f08` |
| `conceptual-components-extraction/project_root/rdf_graph_transformations.py` | `f6d30136a7268361569ba19d834b774781900842` |
| `conceptual-components-extraction/project_root/rdflib_to_networkx_graph.py` | (same commit) |
| `conceptual-components-extraction/project_root/utils.py` | (same commit) |

## What is vendored

Only **step 1** of the upstream pipeline — the part that turns one ontology into
groups of entities. Upstream `project_root/readme.md` numbers seven steps:

| step | script | vendored? | why |
|---|---|---|---|
| 1 | `compute_greedy_modularity_communities.py` | **yes** | produces the per-ontology communities. This *is* the extraction. |
| 2 | `ontology_entities_to_annotatedtext_from_txt.py` | no | turns communities into virtual documents |
| 3 | `ukb` word-sense disambiguation server | no | needs a self-built UKB binary + a 1.6 GB jar hosted on Google Drive |
| 4 | `disambiguation_from_csv.py` | no | consumes the step-3 server |
| 5 | `filter_have_be_enrich_synsetframe.py` | no | Framester frame enrichment |
| 6 | `clustering_*_synsetframe_communities.py` | no | clusters communities **across ontologies** |
| 7 | `clusters_frame_analysis.py` | no | labels the cross-ontology clusters |

Steps 2–7 take the communities of *all* ontologies in a corpus and group them
into a catalogue of *conceptual components* — recurrent structures shared by
several ontologies. That output is a corpus-level catalogue, not a prediction
about which entities of one ontology form one pattern instance, so it is not the
quantity ODP-Reuse scores. The per-ontology grouping that ODP-Reuse *does* score
is exactly the output of step 1, and step 1 has no dependency on steps 2–7.

This is stated in the evaluation section of the paper's use case; it is the
reason the reported figure is labelled "Asprino et al. (step 1)" and never
"Asprino et al." unqualified.

## Patches applied

The upstream code targets Python 3.7, `rdflib==4.2.2`, `networkx==2.4`
(`project_root/environment.yml`). It is run here under Python 3.11,
`rdflib>=7`, `networkx>=3`. Every change is listed:

1. **`import rdfextras` removed** (`rdflib_to_networkx_graph.py`).
   `rdfextras` 0.4 is a Python-2-era package that no longer installs. Nothing in
   the module references it — it is an unused import.
2. **`graphviz` import and the two `*_as_images` functions removed.**
   They render a picture of each community through the Graphviz binary and do
   not affect the returned communities.

   Both are reimplemented outside this directory, so the capability is
   available without the vendored file gaining a dependency on Graphviz:
   `extractbench/render_cc.py` draws a community, and `tools/render_cc_figures.py`
   runs it over a corpus. Three deliberate departures from upstream's drawing
   are listed in that module's docstring — IRIs as node ids rather than local
   names, edge endpoints from the predicate map rather than from splitting a
   local name on `--`, and a layout engine chosen by size. `tools/cc_to_rdf.py`
   likewise supplies upstream's
   `rebuild_intensional_graphs_from_greedy_modularity_communities`, which is
   not vendored either.
3. **File-writing side effects removed** from `print_original_uris_from_communities`
   and `print_recursive_greedy_modularity_communities`; the values they wrote to
   disk are returned instead. The *filter* those writers applied — dropping
   `framester:subsumedUnder` and `framester:extensionallyEquivalent` — is kept,
   because the downstream upstream step reads the written files, so the filtered
   set is the effective output.
4. **Recursion guards added** to `find_all_communities_densitybelowaverage`
   (`_GUARD` blocks, each marked in the source). `networkx>=3`'s
   `greedy_modularity_communities` raises on a graph with no edges, where
   `networkx==2.4` returned a trivial partition; and a Python recursion limit is
   reachable on the deepest ontologies. Both guards terminate the recursion by
   emitting the current node set as one community, which is what the 2.4-era code
   did at the same point.
5. Only the functions on the step-1 execution path are copied. The omitted
   functions in `rdf_graph_transformations.py` are alternative graph
   constructions that `create_owlthing_nodisjoint_class_distinctproperty_nosubsumed`
   — the one the runnable entry point calls — does not use. Diff against the blob
   hashes above to confirm no copied function body was altered.

No SPARQL query text was changed. No parameter was changed. The community
detection algorithm, its recursion criterion, and the graph construction are
upstream's.
