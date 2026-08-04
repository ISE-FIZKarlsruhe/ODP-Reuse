"""
Ground-truth construction in three separately-reported modes.

    iri        an entity counts as reused iff its IRI occurs in both the pattern
               and the ontology. Maximum precision; blind to re-modelled reuse.
    label      an entity counts as reused iff a pattern class and a host entity
               share a NORMALISED LABEL. Captures reuse where the host re-minted
               the pattern under its own namespace.
    iri+label  the union.

The three are never merged silently: every matched entity carries `gt_source`
(`iri` or `label`), so any table can be recomputed for any mode and the strict
subset always remains recoverable.

WHY THERE IS NO SIMILARITY THRESHOLD
------------------------------------
Label matching here is **exact equality on a normalised string**, not similarity.
`_norm_label` splits CamelCase, drops every non-alphanumeric character and
lowercases, so `AgentRole`, `Agent Role`, `agent_role` and `agent-role` all
collapse to `agentrole`, and matching is then `==`.

This is deliberate. Jaccard, edit distance or embedding cosine would each need a
cut-off, and no principled way exists to justify one: the paper's own Limitations
section already argues that "broadening the definition to similarity-based or
structural matching would introduce subjectivity (e.g., at what similarity
threshold does reuse begin?) and undermine the reproducibility central to the
dataset's purpose". Adopting a tuned threshold here would contradict that
position. Exact-normalised equality has no free parameter, is deterministic, and
is reproducible from the definition alone.

The one quantity that *is* continuous -- how much of a pattern was matched -- is
never thresholded either. It is reported as a distribution over coverage levels
(10/30/50/70/90/100%), so the reader chooses the operating point instead of the
author choosing it for them.

TWO PRECISION GUARDS (both parameter-free)
------------------------------------------
Label equality alone over-matches: a class labelled "Event" in an unrelated
ontology is not thereby reusing DUL's Event. Two guards, neither with a tunable
constant:

  1. GENERIC-LABEL BLOCKLIST -- a fixed, tiny set of semantically empty labels
     (`thing`, `entity`, `nothing`, `class`, `concept`) is never matchable.
  2. STRUCTURAL COHERENCE -- an ODP is a CONNECTED fragment, so its reuse should
     be connected too. Among >=2 label-matched host entities, only the largest
     connected component is kept (edges = direct IRI->IRI triples plus one hop
     through a restriction/anonymous node). Scattered coincidental matches are
     dropped. This is a topological criterion, not a tuned one.

Adapted from PatternMiner `rq_0_cache/shared_utils.py` (`_load_gt_from_labels`,
`_norm_label`, `_structurally_coherent`) so that ground truth is identical across
the two papers.

REMAINING RISK, TO BE MEASURED NOT ASSUMED
------------------------------------------
Label matching still admits false positives (homonyms across unrelated domains).
The guards reduce but do not eliminate them, and no automatic check can settle
it. `sample_label_matches()` draws a reproducible random sample of label-only
matches for manual adjudication, so the mode's precision is reported as a
measured number with a confidence interval rather than asserted.
"""

from __future__ import annotations

import random
import re

from rdflib import BNode, Graph, Literal, OWL, RDF, RDFS, URIRef

MODES = ("iri", "label", "iri+label")

# Semantically empty labels that must never license a match.
GENERIC_LABELS: frozenset[str] = frozenset(
    {"thing", "entity", "nothing", "class", "concept", "object", "resource", ""}
)


def norm_label(s: str) -> str:
    """CamelCase -> spaces, then keep alphanumerics only, lowercased.

    `AgentRole`, `Agent Role`, `agent_role`, `agent-role` -> `agentrole`.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(s))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def local_name(iri: str) -> str:
    return iri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def gt_by_iri(host_entities: set[str], pattern_entities: set[str]) -> set[str]:
    """Mode `iri`: exact IRI intersection."""
    return host_entities & pattern_entities


def _host_label_index(graph: Graph, host_entities: set[str]) -> dict[str, set[str]]:
    """normalised label or local name -> host IRIs bearing it."""
    idx: dict[str, set[str]] = {}
    for iri in host_entities:
        for o in graph.objects(URIRef(iri), RDFS.label):
            if isinstance(o, Literal):
                k = norm_label(str(o))
                if k and k not in GENERIC_LABELS:
                    idx.setdefault(k, set()).add(iri)
        k = norm_label(local_name(iri))
        if k and k not in GENERIC_LABELS:
            idx.setdefault(k, set()).add(iri)
    return idx


def structurally_coherent(graph: Graph, iris: set[str]) -> set[str]:
    """Keep only the largest connected component among matched host entities.

    An ODP is a connected fragment; reuse of it should be connected too. Edges are
    direct IRI->IRI triples plus one hop through a blank node (restrictions). A
    single match passes through unchanged. Parameter-free.
    """
    S = list(iris)
    if len(S) <= 1:
        return set(S)
    Sset = set(S)
    adj: dict[str, set[str]] = {e: set() for e in S}
    for a in S:
        for _, _, o in graph.triples((URIRef(a), None, None)):
            if isinstance(o, URIRef) and str(o) in Sset and str(o) != a:
                adj[a].add(str(o))
                adj[str(o)].add(a)
            elif isinstance(o, BNode):
                for _, _, o2 in graph.triples((o, None, None)):
                    if isinstance(o2, URIRef) and str(o2) in Sset and str(o2) != a:
                        adj[a].add(str(o2))
                        adj[str(o2)].add(a)
    seen: set[str] = set()
    best: set[str] = set()
    for start in S:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(adj[u] - seen)
        if len(comp) > len(best):
            best = comp
    return best if len(best) >= 2 else set()


def gt_by_label(host_graph: Graph, host_entities: set[str],
                pattern_graph: Graph) -> set[str]:
    """Mode `label`: normalised-label equality, then both precision guards."""
    idx = _host_label_index(host_graph, host_entities)
    if not idx:
        return set()
    matched: set[str] = set()
    for c, _, _ in pattern_graph.triples((None, RDF.type, OWL.Class)):
        if not isinstance(c, URIRef):
            continue
        keys = {norm_label(local_name(str(c)))}
        for o in pattern_graph.objects(c, RDFS.label):
            if isinstance(o, Literal):
                keys.add(norm_label(str(o)))
        for k in keys:
            if k and k not in GENERIC_LABELS and k in idx:
                matched |= idx[k]
    return structurally_coherent(host_graph, matched)


def ground_truth(host_graph: Graph, host_entities: set[str],
                 pattern_graph: Graph, pattern_entities: set[str],
                 mode: str = "iri") -> tuple[set[str], dict[str, str]]:
    """Return (matched host entities, {entity: gt_source}).

    gt_source is `iri` when the entity matched by IRI and `label` otherwise, so a
    mixed-mode result can always be decomposed back into its parts.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    by_iri = gt_by_iri(host_entities, pattern_entities) if mode != "label" else set()
    by_label = (gt_by_label(host_graph, host_entities, pattern_graph)
                if mode != "iri" else set())

    source = {e: "iri" for e in by_iri}
    for e in by_label:
        source.setdefault(e, "label")
    return (by_iri | by_label), source


def sample_label_matches(pairs: list[dict], n: int = 50, seed: int = 0) -> list[dict]:
    """Reproducible sample of LABEL-ONLY matches for manual adjudication.

    Precision of the label mode cannot be established automatically. Draw this
    sample, adjudicate each row by hand, and report precision with a Wilson
    interval. `pairs` rows need `n_reused_label_only` > 0.
    """
    cands = [p for p in pairs if int(p.get("n_reused_label_only") or 0) > 0]
    rng = random.Random(seed)
    rng.shuffle(cands)
    out = []
    for p in cands[:n]:
        row = dict(p)
        row["manual_verdict"] = ""       # fill in: reuse | not_reuse | unclear
        row["manual_note"] = ""
        out.append(row)
    return out
