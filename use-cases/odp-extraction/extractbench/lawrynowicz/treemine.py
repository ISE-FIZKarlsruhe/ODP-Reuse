"""Frequent, unordered, **induced** subtree mining, per Lawrynowicz et al. §4.3-4.4.

WHAT THE PAPER SPECIFIES
------------------------
* Algorithm: SLEUTH (Zaki 2005), reimplemented by the authors in Java as
  FF-SLEUTH ("Family of Forests SLEUTH"). The published source address,
  ``https://bitbucket.org/leolod/jsleuth``, returns HTTP 404: the code is not
  available, so this is a reimplementation from the paper.
* Subtree notion: **unordered and induced**. Embedded-but-not-induced subtrees
  are explicitly discarded, because they license axiom claims the source axioms
  do not support: from ``Heart SubClassOf hasTopology only Tubular`` and
  ``Heart SubClassOf hasFeature some not Tubular`` the embedded subtree
  ``Heart SubClassOf Tubular`` "is not justified by the original axioms".
* Support, single forest: the number of trees (axioms) of that forest in which
  the subtree occurs -- occurrences within one tree are counted once. Relative
  threshold 1%.
* Support, family of forests: the number of *forests* (ontologies) containing at
  least one occurrence, no matter how many axioms match inside each. Threshold
  4 ontologies (the paper's "1% of 331 non-empty forests").
* Condensation: **maximal** frequent subtrees only -- "frequent subtrees for
  which none of their proper supertrees is frequent". On the paper's corpus this
  cut 1,935,735 frequent subtrees to 640,075 maximal ones.
* Structural filter: keep only subtrees containing a ``SubClassOf`` or
  ``EquivalentTo`` node, so that a decoded pattern is a whole axiom pattern and
  not a class expression of unknown nesting depth.

HOW THIS IMPLEMENTATION WORKS, AND WHY IT IS EQUIVALENT
-------------------------------------------------------
SLEUTH enumerates candidates by rightmost extension and tests occurrences with
scope lists. This module enumerates the same search space by a different route
that is easier to verify, because it exploits a property specific to *induced*
subtrees of rooted trees:

    An induced-subtree embedding maps the parent of a node to the parent of its
    image. Its image is therefore a node set that is closed under "parent, if
    the parent is also in the set" and connected. Conversely, any such node set
    induces exactly one subtree. **Occurrence and image set are in bijection**,
    and the pattern is determined by the image set.

So the enumeration is over connected, parent-closed node subsets of each axiom
tree, level-wise by size: size-1 subsets are the individual nodes; a size-(k+1)
subset is a size-k subset plus one node whose parent is already inside. Subsets
are grouped by the canonical form of the tree they induce, which is the pattern.
Anti-monotonicity holds -- removing a leaf from a parent-closed set leaves a
parent-closed set -- so pruning infrequent patterns before extending is both
sound and complete, exactly as in the level-wise SLEUTH search.

Canonical form of an unordered labelled rooted tree: ``canon(v) = (label(v),
sorted(canon(c) for c in children(v)))``. Two trees are isomorphic as unordered
trees iff their canonical forms are equal.

Because every occurrence is retained, the ontology and axiom a pattern matches
are known without a second matching pass. That is what lets the benchmark turn a
mined pattern back into a set of entities in a specific ontology.

DEVIATIONS, ALL DELIBERATE AND ALL BOUNDED
------------------------------------------
``max_size``   The paper imposes no size cap; sizes are emergent. It reports
               observed sizes of 2-12 for the cross-corpus run (and up to 43 for
               single huge ontologies). A cap is applied here so that a
               pathological ontology cannot stall the sweep. Its value is
               recorded in the results and the number of patterns that hit it is
               reported, so truncation is never silent.
``max_occ``    Occurrences kept per (pattern, tree). Support counts transactions,
               not occurrences, so capping cannot change which patterns are
               frequent unless the cap is hit on *every* occurrence of a tree,
               which cannot happen because at least one is always kept. It
               bounds memory only.
"""

from __future__ import annotations

from collections import defaultdict

from .axiom_trees import AxiomTree, EQUIVALENT_TO, SUBCLASS_OF, T_CC

DEFAULT_MAX_SIZE = 12          # the paper's observed cross-corpus range is 2-12
DEFAULT_MAX_OCC = 400          # per (pattern, tree); memory bound only


# ----------------------------------------------------------------------
# flat representation of one axiom tree
# ----------------------------------------------------------------------

class FlatTree:
    """An AxiomTree flattened to arrays, with preorder ids."""

    __slots__ = ("labels", "parent", "children", "source", "n")

    def __init__(self, tree: AxiomTree):
        self.labels: list[tuple[str, str]] = []
        self.parent: list[int] = []
        self.children: list[list[int]] = []
        self.source = tree
        stack = [(tree.root, -1)]
        while stack:
            node, par = stack.pop()
            i = len(self.labels)
            self.labels.append(node.label)
            self.parent.append(par)
            self.children.append([])
            if par >= 0:
                self.children[par].append(i)
            for c in reversed(node.children):
                stack.append((c, i))
        self.n = len(self.labels)

    def canon(self, nodes: frozenset[int]):
        """Canonical form of the subtree induced by ``nodes``."""
        root = next(i for i in nodes if self.parent[i] not in nodes)
        return self._canon(root, nodes)

    def _canon(self, i: int, nodes: frozenset[int]):
        kids = sorted(self._canon(c, nodes) for c in self.children[i] if c in nodes)
        return (self.labels[i], tuple(kids))


def canon_labels(canon) -> list[tuple[str, str]]:
    """Every node label of a canonical form, in canonical preorder."""
    out = [canon[0]]
    for kid in canon[1]:
        out.extend(canon_labels(kid))
    return out


def canon_size(canon) -> int:
    return 1 + sum(canon_size(k) for k in canon[1])


def is_axiom_pattern(canon) -> bool:
    """The paper's filter: the subtree must contain a SubClassOf/EquivalentTo node."""
    return any(lbl == (T_CC, SUBCLASS_OF) or lbl == (T_CC, EQUIVALENT_TO)
               for lbl in canon_labels(canon))


# ----------------------------------------------------------------------
# the miner
# ----------------------------------------------------------------------

class MinedPattern:
    """One frequent subtree, with every occurrence that produced it."""

    __slots__ = ("canon", "size", "occurrences", "maximal")

    def __init__(self, canon, size: int):
        self.canon = canon
        self.size = size
        # forest_id -> tree_index -> set of image node sets. A set, not a list:
        # the same image is reached once per removable leaf, so the dedupe is on
        # the inner loop and must be O(1).
        self.occurrences: dict[int, dict[int, set[frozenset[int]]]] = defaultdict(
            lambda: defaultdict(set))
        self.maximal = True

    def support_forests(self) -> int:
        return len(self.occurrences)

    def support_trees(self) -> int:
        return sum(len(t) for t in self.occurrences.values())

    def n_occurrences(self) -> int:
        return sum(len(v) for t in self.occurrences.values() for v in t.values())


def mine(forests: list[list[FlatTree]], min_support: int,
         support_unit: str = "forest",
         max_size: int = DEFAULT_MAX_SIZE,
         max_occ: int = DEFAULT_MAX_OCC,
         progress=None) -> tuple[list[MinedPattern], dict]:
    """Mine maximal frequent induced unordered subtrees.

    ``support_unit`` is ``"forest"`` for the family-of-forests run (count
    ontologies, the paper's cross-corpus setting) or ``"tree"`` for a
    single-forest run (count axioms).

    Returns ``(maximal_patterns, stats)``. ``stats`` records the level-wise
    counts and how many patterns were cut off by ``max_size``, so a truncated
    sweep is visible in the results rather than silent.
    """
    def support(p: MinedPattern) -> int:
        return p.support_forests() if support_unit == "forest" else p.support_trees()

    # ---- level 1 -----------------------------------------------------
    level: dict = {}
    for fid, forest in enumerate(forests):
        for tid, ft in enumerate(forest):
            for i in range(ft.n):
                c = ((ft.labels[i]), ())
                p = level.get(c)
                if p is None:
                    p = level[c] = MinedPattern(c, 1)
                occ = p.occurrences[fid][tid]
                if len(occ) < max_occ:
                    occ.add(frozenset((i,)))

    level = {c: p for c, p in level.items() if support(p) >= min_support}
    all_patterns: dict = dict(level)
    stats = {"levels": [len(level)],
             "occurrences": [sum(p.n_occurrences() for p in level.values())],
             "hit_max_size": 0,
             "min_support": min_support, "support_unit": support_unit,
             "max_size": max_size}

    # ---- level k -> k+1 ----------------------------------------------
    size = 1
    while level and size < max_size:
        # The same (k+1)-image is reached once per removable leaf of the pattern
        # it extends, so images are deduped per tree before any canonical form
        # is computed -- canonicalisation is the inner-loop cost.
        seen: dict[tuple[int, int], set[frozenset[int]]] = defaultdict(set)
        nxt: dict = {}
        for p in level.values():
            for fid, per_tree in p.occurrences.items():
                for tid, images in per_tree.items():
                    ft = forests[fid][tid]
                    seen_here = seen[(fid, tid)]
                    for img in images:
                        for i in img:
                            for c in ft.children[i]:
                                if c in img:
                                    continue
                                new_img = img | {c}
                                if new_img in seen_here:
                                    continue
                                seen_here.add(new_img)
                                cc = ft.canon(new_img)
                                np_ = nxt.get(cc)
                                if np_ is None:
                                    np_ = nxt[cc] = MinedPattern(cc, size + 1)
                                occ = np_.occurrences[fid][tid]
                                if len(occ) < max_occ:
                                    occ.add(new_img)
        nxt = {c: p for c, p in nxt.items() if support(p) >= min_support}

        # A frequent (k+1)-pattern makes every k-pattern obtainable from it by
        # removing one leaf non-maximal.
        for cc in nxt:
            for parent_canon in _leaf_removals(cc):
                pp = level.get(parent_canon)
                if pp is not None:
                    pp.maximal = False

        stats["levels"].append(len(nxt))
        stats["occurrences"].append(sum(p.n_occurrences() for p in nxt.values()))
        if progress:
            progress(f"    size {size + 1}: {len(nxt)} frequent patterns, "
                     f"{stats['occurrences'][-1]} occurrences")
        all_patterns.update(nxt)
        level = nxt
        size += 1

    if level and size >= max_size:
        stats["hit_max_size"] = len(level)

    maximal = [p for p in all_patterns.values() if p.maximal]
    stats["n_frequent"] = len(all_patterns)
    stats["n_maximal"] = len(maximal)
    return maximal, stats


def _leaf_removals(canon):
    """Canonical forms obtained by deleting one leaf of ``canon``."""
    out = set()
    label, kids = canon
    for idx, kid in enumerate(kids):
        rest = kids[:idx] + kids[idx + 1:]
        if not kid[1]:                       # kid is a leaf: delete it
            out.add((label, tuple(sorted(rest))))
        else:                                # recurse into the kid
            for sub in _leaf_removals(kid):
                out.add((label, tuple(sorted(rest + (sub,)))))
    return out
