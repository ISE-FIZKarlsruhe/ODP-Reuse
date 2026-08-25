"""Render each conceptual-components community as a figure.

Upstream ships this capability as
``print_recursive_greedy_modularity_communities_as_images`` and
``print_norecursive_greedy_modularity_communities_as_images`` in
``compute_greedy_modularity_communities.py``. Both were dropped when that file
was vendored (PROVENANCE.md, PATCH 2) because they only draw pictures and do
not affect the returned communities. This module restores the capability
against the same Graphviz backend, reading the same two structures upstream's
renderer reads.

WHAT A COMMUNITY LOOKS LIKE
---------------------------
Step 1 clusters the *reified* intensional graph, in which every triple
``s p o`` of the intensional graph has been turned into a node
``http://edge.example/<s>--<p>--<o>`` sitting between ``s`` and ``o``. A
community is therefore a mixed bag of ordinary entity nodes and edge nodes, and
drawing it means undoing the reification: an edge node becomes an arrow from
``s`` to ``o`` labelled ``p``, and everything else becomes a plain node.

THREE DEPARTURES FROM UPSTREAM'S RENDERER, ALL DELIBERATE
---------------------------------------------------------
1. **Nodes are identified by full IRI, labelled by local name.** Upstream calls
   ``getID`` to produce the Graphviz node id, so two entities from different
   namespaces that share a local name -- ``dul:Event`` and ``ex:Event``, say --
   are drawn as one node and the community looks smaller and better connected
   than it is. The full IRI is used as the id here and the local name only as
   the label, so the drawing has exactly as many nodes as the community does.
2. **Edge endpoints come from the predicate map, not from splitting the local
   name.** Upstream recovers ``(s, p, o)`` as ``getID(edge_iri).split("--")``,
   which mis-splits whenever a local name itself contains ``--``. The map
   ``create_class_distinctproperty_graph`` already returns holds the original
   three IRIs, so it is used, with upstream's split as the fallback.
3. **PNG, and a layout engine chosen by size.** Upstream renders with the
   Graphviz default format and always with ``dot``. Communities here run from 2
   to 187 entities; ``dot`` is right for the small ones and lays the large ones
   out in an unreadably wide band, so anything above ``SFDP_ABOVE`` nodes is
   rendered with ``sfdp`` instead.

Upstream's ``subsumedUnder`` / ``extensionallyEquivalent`` branches are kept
below for fidelity, but they cannot fire in this pipeline: the graph that gets
clustered is built by
``create_owlthing_nodisjoint_class_distinctproperty_nosubsumed``, which drops
both relations before reification.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_USECASE = Path(__file__).resolve().parent.parent
if str(_USECASE) not in sys.path:
    sys.path.insert(0, str(_USECASE))

from third_party.conceptual_components.compute_greedy_modularity_communities import (  # noqa: E402
    print_recursive_greedy_modularity_communities,
)
from third_party.conceptual_components.rdf_graph_transformations import (  # noqa: E402
    create_owlthing_nodisjoint_class_distinctproperty_nosubsumed,
)
from third_party.conceptual_components.rdflib_to_networkx_graph import (  # noqa: E402
    from_rdflib_to_networkx_graph,
)
from third_party.conceptual_components.utils import getID  # noqa: E402

EDGE_PREFIX = "http://edge.example/"
SUBSUMED = "https://w3id.org/framester/schema/subsumedUnder"
EQUIVALENT = "https://w3id.org/framester/schema/extensionallyEquivalent"
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"

# Above this many drawn nodes or arcs, `dot` produces a very wide, thin band;
# a force-directed layout keeps the graph on a page that can be read.
#
# `fdp`, not `sfdp`: sfdp always calls remove_overlap, which needs a
# triangulation library that Graphviz is not always built with -- on such a
# build every sfdp render fails outright, whatever `overlap` is set to. fdp
# needs no triangulation and lays out graphs of this size just as well.
LARGE_ABOVE = 40
BIG_ENGINE = "fdp"
SFDP_ABOVE = LARGE_ABOVE          # retained: older callers import this name
LABEL_MAX = 34
# Property names printed on one arc before the rest become a "+N more" count.
PARALLEL_LABELS = 6

_WS = re.compile(r"[\s ]+")


def communities_with_edges(graph):
    """Step 1, keeping what the renderer needs.

    Returns ``(communities, pred_map)`` where each community is the list of
    node IRIs upstream's clustering produced -- edge nodes *not* expanded --
    and ``pred_map`` maps an edge node's IRI to its ``[s, p, o]`` originals.

    ``extractbench.asprino.communities`` calls the same two functions and then
    expands the edge nodes away, which is right for scoring and useless for
    drawing; this keeps them.
    """
    built = create_owlthing_nodisjoint_class_distinctproperty_nosubsumed(graph)
    property_graph, pred_map = built[0][0], built[0][1]
    if len(property_graph) == 0:
        return [], {}
    nx_graph = from_rdflib_to_networkx_graph(property_graph)
    return print_recursive_greedy_modularity_communities(nx_graph), pred_map


def _triple(uri: str, pred_map: dict) -> list[str] | None:
    """The ``[s, p, o]`` an edge node stands for, or None if it is not one."""
    if uri in pred_map:
        return list(pred_map[uri])
    if uri.startswith(EDGE_PREFIX):
        parts = getID(uri).split("--")
        return parts if len(parts) == 3 else None
    return None


def _label(uri: str) -> str:
    """Local name, whitespace-flattened and clipped so a node stays a node."""
    text = _WS.sub(" ", getID(uri) or uri).strip() or uri
    return text if len(text) <= LABEL_MAX else text[: LABEL_MAX - 1] + "…"


def community_content(uris, pred_map):
    """Split a community into the triples and the standalone nodes it draws.

    Returned as ``(triples, singles)``: ``triples`` is a list of
    ``(s, p, o, style)`` with ``style`` in ``{"plain", "subsumed",
    "equivalent"}``, and ``singles`` the IRIs that take part in no triple of
    this community. Both are what ends up in the figure, so the text listing
    written beside it describes exactly the drawing.

    A community holds an entity node in its own right *as well as* the edge
    nodes that entity sits between, so a member can be both a plain node and an
    endpoint. Such a member is not standalone -- it is already drawn, and
    already in the RDF -- and is therefore removed from ``singles`` rather than
    reported as taking part in nothing.
    """
    triples, plain = [], []
    for uri in uris:
        if SUBSUMED in uri or EQUIVALENT in uri:
            t = _triple(uri, pred_map)
            if t:
                triples.append((t[0], t[1], t[2],
                                "subsumed" if SUBSUMED in uri else "equivalent"))
            continue
        t = _triple(uri, pred_map)
        if t:
            triples.append((t[0], t[1], t[2], "plain"))
        else:
            plain.append(uri)
    endpoints = {s for s, _, _, _ in triples} | {o for _, _, o, _ in triples}
    singles = [u for u in plain if u not in endpoints]
    return triples, singles


def group_edges(triples):
    """Collapse relations sharing an ordered pair of endpoints into one arc.

    Necessary, not cosmetic. The pipeline gives every property that declares
    neither domain nor range ``owl:Thing`` for both, so an ontology of bare
    object properties -- ``cito`` is the extreme, with 98 of them -- yields one
    community whose every relation runs from ``owl:Thing`` to ``owl:Thing``.
    Drawn one arc apiece that is 98 stacked self-loops on a single node, a
    figure some 6,000 pixels tall in which nothing can be read. Drawn as one
    arc carrying the property names it is the same fact, legibly.

    Returns ``(groups, n_collapsed)`` with each group ``(s, o, style, [p, ...])``
    in first-appearance order, and ``n_collapsed`` the number of relations that
    share an arc with another.
    """
    groups: dict[tuple[str, str, str], list[str]] = {}
    order: list[tuple[str, str, str]] = []
    for s, p, o, style in triples:
        key = (s, o, style)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(p)
    collapsed = sum(len(v) - 1 for v in groups.values() if len(v) > 1)
    return [(s, o, style, groups[(s, o, style)]) for s, o, style in order], collapsed


def _edge_label(preds: list[str]) -> str:
    """One arc's label: the property names, capped, stacked one per line."""
    names = [_label(p) for p in preds]
    if len(names) <= PARALLEL_LABELS:
        return "\\n".join(names)
    return ("\\n".join(names[:PARALLEL_LABELS])
            + f"\\n+{len(names) - PARALLEL_LABELS} more")


def render_community(uris, pred_map, out_stem: Path, **kwargs) -> Path | None:
    """Draw one community. Returns the written file, or None if it drew nothing.

    ``out_stem`` is a path without extension, as Graphviz expects.
    """
    triples, singles = community_content(uris, pred_map)
    return render_graph(triples, singles, out_stem, **kwargs)


def render_graph(triples, singles, out_stem: Path, title: str = "",
                 fmt: str = "png", dpi: int = 150,
                 keep_source: bool = False, highlight=None) -> Path | None:
    """Draw any entity graph: ``triples`` of ``(s, p, o, style)`` plus the IRIs
    in ``singles``, which take part in no triple.

    Kept separate from ``render_community`` so that a cluster from any other
    extraction method -- whose members arrive as a bare list of IRIs with the
    relations recovered from an extracted module rather than from reified edge
    nodes -- is drawn by exactly the same code, and so is comparable to a
    conceptual-components figure at a glance.

    ``highlight`` is the set of IRIs that *are* the cluster. A module drawn for
    a cluster necessarily brings in its neighbourhood -- every subclass of a
    cluster member arrives with it -- so a five-entity cluster can draw twenty
    nodes, and without the distinction the figure cannot say which five were
    predicted. Highlighted entities keep the solid fill; everything drawn only
    because it touches one is muted. Pass None (the default, and what the
    conceptual-components figures use) when every node is a member.
    """
    from graphviz import Digraph

    if not triples and not singles:
        return None

    groups, _collapsed = group_edges(triples)
    drawn = {s for s, _, _, _ in triples} | {o for _, _, o, _ in triples}
    drawn |= set(singles)
    # Both counts matter: `dot` handles many nodes badly *and* many arcs badly,
    # and a community can have few of one and many of the other.
    engine = (BIG_ENGINE if len(drawn) > LARGE_ABOVE or len(groups) > LARGE_ABOVE
              else "dot")

    # An IRI cannot be a Graphviz node id. `graphviz.quoting.quote_edge`
    # partitions an endpoint on the first ':' to preserve DOT's `node:port`
    # syntax, so "https://example.org/Foo" reaches the layout engine as node
    # `https` with port `//example.org/Foo` -- every IRI in a community
    # collapses onto one node named `https`, and Graphviz warns once per edge.
    # Each IRI therefore gets an opaque surrogate id and carries its IRI in the
    # label and tooltip instead.
    nid = {iri: f"n{i}" for i, iri in enumerate(sorted(drawn))}

    g = Digraph(name=out_stem.name, engine=engine, format=fmt)
    g.attr(rankdir="LR", dpi=str(dpi), bgcolor="white",
           splines="true", overlap="false", nodesep="0.28", ranksep="0.55",
           pad="0.3", fontname="Helvetica")
    # Say so on the figure when arcs were collapsed, so the drawing is never
    # read as showing fewer relations than the community holds.
    caption = title
    if _collapsed:
        note = (f"{_collapsed} further relation"
                f"{'s' if _collapsed > 1 else ''} share an arc shown here")
        caption = f"{title}\\n{note}" if title else note
    if caption:
        g.attr(label=caption, labelloc="t", labeljust="c",
               fontsize="13", fontname="Helvetica-Bold", fontcolor="#1a1a1a")
    g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica",
           fontsize="10", color="#3a6ea5", fillcolor="#dbe9f6",
           fontcolor="#10243a", penwidth="1.1", margin="0.10,0.055")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#5a5a5a",
           fontcolor="#333333", arrowsize="0.7", penwidth="1.0")

    # owl:Thing is injected by the pipeline wherever a property declares no
    # domain or range, so it is scaffolding rather than something the ontology
    # says. Drawn greyed out so it is not read as a modelling choice.
    seeds = set(highlight) if highlight is not None else None
    for iri in sorted(drawn):
        if iri == OWL_THING:
            g.node(nid[iri], label="owl:Thing", style="rounded,filled,dashed",
                   color="#9a9a9a", fillcolor="#f2f2f2", fontcolor="#555555",
                   tooltip=iri)
        elif seeds is not None and iri not in seeds:
            g.node(nid[iri], label=_label(iri), tooltip=f"{iri} (context)",
                   style="rounded", color="#a8bcd0", fillcolor="white",
                   fontcolor="#5c6b7a", penwidth="0.9")
        else:
            g.node(nid[iri], label=_label(iri), tooltip=iri)

    for s, o, style, preds in groups:
        tip = " | ".join(preds)
        if style == "subsumed":
            g.edge(nid[s], nid[o], arrowhead="empty", color="#3a6ea5",
                   tooltip=tip)
        elif style == "equivalent":
            g.edge(nid[s], nid[o], label="extensionallyEquivalent",
                   arrowhead="none", style="dashed", tooltip=tip)
        else:
            g.edge(nid[s], nid[o], label=_edge_label(preds), tooltip=tip,
                   penwidth="1.6" if len(preds) > 1 else "1.0")

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    # Not `cleanup=True`: graphviz deletes the .gv source *after* writing the
    # figure and lets the failure propagate, so a virus scanner or search
    # indexer holding the source for a moment turns a rendered figure into a
    # raised exception and a lost record of it. Delete it separately, and treat
    # failing to delete a temporary file as the non-event it is.
    try:
        produced = Path(g.render(filename=str(out_stem), cleanup=False))
    except Exception:
        # A Graphviz build can lack what a layout engine needs. Losing the
        # figure over that is worse than losing the layout, so retry with
        # `dot`, which every build has.
        if engine == "dot":
            raise
        g.engine = "dot"
        produced = Path(g.render(filename=str(out_stem), cleanup=False))
    if not keep_source:
        try:
            out_stem.unlink()
        except OSError:
            pass
    return produced


def community_listing(triples, singles) -> str:
    """Upstream's per-community text block, over full IRIs rather than local
    names, so the listing identifies entities the drawing can only label."""
    lines = []
    for s, p, o, style in triples:
        marker = {"subsumed": " [subsumedUnder]",
                  "equivalent": " [extensionallyEquivalent]"}.get(style, "")
        lines.append(f"\t{s} {p} {o}{marker}")
    for iri in singles:
        lines.append(f"\t{iri}")
    return "\n".join(lines)
