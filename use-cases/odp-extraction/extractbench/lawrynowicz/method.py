"""Reading Lawrynowicz et al.'s mined patterns back onto one ontology.

The method is corpus-level: it mines a catalogue of axiom patterns from a family
of forests, then per ontology mines class frame patterns over those axiom
patterns. Neither output is a set of entities. This module fixes, once and
visibly, how a mined pattern becomes the thing ODP-Reuse scores.

TWO GRANULARITIES ARE REPORTED, MATCHING THE PAPER'S TWO OUTPUT LEVELS
---------------------------------------------------------------------
``axiom_pattern`` (``law_ap``)
    One predicted cluster per maximal frequent axiom pattern that occurs in the
    ontology. Its members are the entities of the axioms it matches there.
``class_frame`` (``law_cfp``)
    One predicted cluster per class frame pattern mined for that ontology. Its
    members are the entities of the axioms matched, in that ontology, by the
    CFP's member patterns on the classes that support the CFP. This is the level
    the paper calls an emerging design pattern, and is the primary figure.

WHICH ENTITIES BELONG TO A MATCH
--------------------------------
``binding="axiom"`` (default)
    Every entity of the matched axiom. An axiom pattern with variables is an
    axiom shape; its instance in an ontology is a concrete axiom, and the
    entities bound to the variables are part of that instance.
``binding="ground"``
    Only the entities named in the pattern itself, i.e. those at the matched
    nodes. This scores the ground fragments (the paper's APNV, "ontology
    fragment") and ignores variable bindings.
Both are computed; the runner reports the first and records the second, so the
choice is visible rather than assumed.

FITTING
-------
``fit_corpus`` runs the family-of-forests mining over every ontology in the
evaluation population at once -- the paper's cross-corpus configuration, support
counted in ontologies, threshold 4. The mined catalogue is then shared by every
ontology, which is the point: a pattern is a regularity across ontologies, and
an ontology mined alone can only produce regularities across its own axioms.
Passing ``support_unit="tree"`` instead reproduces the paper's per-ontology
configuration (support counted in axioms, relative threshold 1%).

Nothing in the fit consults the ground truth. The corpus of *ontologies* is
shared with the evaluation, as it must be for an unsupervised corpus-level
method, but no pattern label, no ontology-pattern pair and no entity of the
answer key is read at any point.
"""

from __future__ import annotations

from collections import defaultdict

from .axiom_trees import forest
from .cfp import maximal_frequent_itemsets
from .treemine import FlatTree, is_axiom_pattern, mine

DEFAULT_MIN_SUPPORT = 4        # paper: 4 forests for the cross-corpus run
DEFAULT_SIGMA_CF = 4           # paper: 4 transactions for the itemset run


class Fitted:
    """The mined catalogue plus, per ontology, everything needed to score it."""

    def __init__(self):
        self.stats: dict = {}
        # ontology_key -> granularity -> list of entity sets
        self.clusters: dict[str, dict[str, list[set[str]]]] = {}


def fit_corpus(graphs: dict[str, object],
               min_support: int = DEFAULT_MIN_SUPPORT,
               sigma_cf: int = DEFAULT_SIGMA_CF,
               support_unit: str = "forest",
               max_size: int = 12,
               relative_support: float | None = None,
               pre_encoded: bool = False,
               progress=None) -> Fitted:
    """Mine the corpus and derive per-ontology clusters at both granularities.

    ``graphs`` maps an ontology key to its parsed rdflib graph. With
    ``support_unit="forest"`` one catalogue is mined over all of them; with
    ``"tree"`` each ontology is mined on its own and ``relative_support``
    (the paper's 1%) sets its threshold.

    ``pre_encoded=True`` means the values are already lists of ``FlatTree``.
    A caller that parses a large corpus one ontology at a time uses this to
    hand over the forests and release each graph immediately, instead of
    keeping every graph alive until the fit.
    """
    keys = list(graphs)
    forests: list[list[FlatTree]] = []
    for i, k in enumerate(keys):
        if pre_encoded:
            forests.append(graphs[k])          # already FlatTrees
        else:
            forests.append([FlatTree(t) for t in forest(graphs[k])])
        if progress and (i + 1) % 50 == 0:
            progress(f"  encoded {i + 1}/{len(keys)} forests")

    out = Fitted()

    if support_unit == "tree":
        # Paper's per-ontology configuration: mine each forest alone, relative
        # support threshold (1%) inside that forest.
        rel = relative_support if relative_support is not None else 0.01
        agg_stats = {"levels": [], "n_frequent": 0, "n_maximal": 0,
                     "hit_max_size": 0, "min_support": f"rel {rel}",
                     "support_unit": "tree", "max_size": max_size}
        for fid, key in enumerate(keys):
            f = forests[fid]
            if not f:
                out.clusters[key] = {"axiom_pattern": [], "class_frame": [],
                                     "axiom_pattern_ground": [],
                                     "class_frame_ground": []}
                continue
            ms = max(1, int(rel * len(f)))
            pats, st = mine([f], ms, "tree", max_size=max_size)
            pats = [p for p in pats if is_axiom_pattern(p.canon)]
            agg_stats["n_frequent"] += st["n_frequent"]
            agg_stats["n_maximal"] += len(pats)
            agg_stats["hit_max_size"] += st["hit_max_size"]
            out.clusters[key] = _clusters_for(pats, [f], 0, key, sigma_cf)
        out.stats = agg_stats
        return out

    if progress:
        progress(f"  mining family of forests: {len(forests)} ontologies, "
                 f"{sum(len(f) for f in forests)} axiom trees, minsup={min_support}")
    patterns, stats = mine(forests, min_support, "forest", max_size=max_size,
                           progress=progress)
    patterns = [p for p in patterns if is_axiom_pattern(p.canon)]
    stats["n_axiom_patterns"] = len(patterns)
    out.stats = stats
    if progress:
        progress(f"  {stats['n_frequent']} frequent, {stats['n_maximal']} maximal, "
                 f"{len(patterns)} axiom patterns")

    for fid, key in enumerate(keys):
        out.clusters[key] = _clusters_for(patterns, forests, fid, key, sigma_cf)
    return out


def _clusters_for(patterns, forests, fid: int, key: str,
                  sigma_cf: int) -> dict[str, list[set[str]]]:
    """Per-ontology clusters at both granularities and both bindings."""
    f = forests[fid] if fid < len(forests) else []
    # pattern index -> tree indices it matches in this ontology
    hits: dict[int, set[int]] = {}
    ground: dict[int, dict[int, set[str]]] = defaultdict(dict)
    for pi, p in enumerate(patterns):
        per_tree = p.occurrences.get(fid)
        if not per_tree:
            continue
        hits[pi] = set(per_tree)
        for tid, images in per_tree.items():
            ft = f[tid]
            named: set[str] = set()
            for img in images:
                for i in img:
                    typ, name = ft.labels[i]
                    if typ in ("C", "OP", "DP", "I", "LHS"):
                        named.add(name)
            ground[pi][tid] = named

    ap: list[set[str]] = []
    ap_ground: list[set[str]] = []
    for pi, tids in hits.items():
        ents: set[str] = set()
        gents: set[str] = set()
        for tid in tids:
            ents |= set(f[tid].source.entities)
            gents |= ground[pi].get(tid, set())
        if ents:
            ap.append(ents)
        if gents:
            ap_ground.append(gents)

    # ---- class frame patterns (paper §4.6) ---------------------------
    # transactions: named classes on the LHS of a mined axiom.
    class_items: dict[str, set[int]] = defaultdict(set)
    class_trees: dict[str, set[int]] = defaultdict(set)
    for tid, ft in enumerate(f):
        class_trees[ft.source.lhs].add(tid)
    for pi, tids in hits.items():
        for tid in tids:
            class_items[f[tid].source.lhs].add(pi)

    tx_classes = [c for c in class_trees if class_items.get(c)]
    transactions = [frozenset(class_items[c]) for c in tx_classes]
    cfps = maximal_frequent_itemsets(transactions, sigma_cf)

    cf: list[set[str]] = []
    cf_ground: list[set[str]] = []
    for itemset, _n in cfps:
        ents: set[str] = set()
        gents: set[str] = set()
        for c, tx in zip(tx_classes, transactions):
            if not itemset <= tx:
                continue
            for pi in itemset:
                for tid in hits[pi] & class_trees[c]:
                    ents |= set(f[tid].source.entities)
                    gents |= ground[pi].get(tid, set())
        if ents:
            cf.append(ents)
        if gents:
            cf_ground.append(gents)

    return {"axiom_pattern": ap, "class_frame": cf,
            "axiom_pattern_ground": ap_ground, "class_frame_ground": cf_ground}


def extract(fitted: Fitted, key: str, entities: list[str],
            granularity: str = "class_frame") -> dict[int, set[str]]:
    """Predicted clusters for one ontology, in the runner's cluster format.

    Clusters are restricted to ``entities`` -- the same universe the embedding
    baselines cluster over and the ground truth lives in -- so precision is not
    penalised for a disagreement about what counts as an entity.
    """
    universe = set(entities)
    groups = fitted.clusters.get(key, {}).get(granularity, [])
    out: dict[int, set[str]] = {}
    for grp in groups:
        m = grp & universe
        if m:
            out[len(out)] = m
    return out
