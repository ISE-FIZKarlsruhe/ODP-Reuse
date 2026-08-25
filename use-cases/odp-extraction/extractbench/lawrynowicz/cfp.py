"""Class frame patterns, per Lawrynowicz et al. §4.6.

The paper's stage 2, restated:

    "each frequent axiom pattern Q_a represents a transaction item, and each
    named class A in the ontology O appearing on the left-hand side of any
    SubClassOf or EquivalentTo axiom has an associated transaction t_A. Each
    transaction t_i is represented by the set of all frequent axiom patterns
    (items) whose right-hand side matches the SubClassOf or EquivalentTo
    ontology axioms having A on the left-hand side."

    "we apply an off-the-shelf frequent itemset mining algorithm. The result of
    the algorithm will be a set of itemsets, which correspond to class frame
    patterns." -- "we used Orange3-Associate to mine maximal frequent itemsets
    using 4 as the minimum support sigma_CF threshold."

A *class frame pattern* is therefore a set of axiom patterns that co-occur on at
least ``sigma_CF`` named classes of one ontology, and it is what the paper calls
an (emerging) design pattern. CFP mining is per-ontology: "On average, there are
16.3 CFPs per ontology."

Orange3-Associate's ``frequent_itemsets`` is an FP-growth implementation. This
module implements maximal frequent itemset mining directly, because the input
here is a few hundred transactions over a few thousand items and an FP-growth
dependency would buy nothing but a version to pin. Maximality is the paper's
condensation, and is applied here to the same definition: an itemset is maximal
when no proper superset is frequent.
"""

from __future__ import annotations

from collections import defaultdict


def maximal_frequent_itemsets(transactions: list[frozenset[int]],
                              min_support: int,
                              max_len: int = 8) -> list[tuple[frozenset[int], int]]:
    """Maximal frequent itemsets, level-wise (Apriori enumeration).

    ``max_len`` caps the itemset length. The paper imposes none; the cap is a
    guard against a degenerate ontology in which hundreds of classes carry an
    identical axiom set, where the maximal itemset is unique but the level-wise
    enumeration below it is exponential. Callers report when it is reached.
    """
    if not transactions:
        return []

    counts: dict[int, int] = defaultdict(int)
    for t in transactions:
        for item in t:
            counts[item] += 1

    level: dict[frozenset[int], int] = {
        frozenset((i,)): c for i, c in counts.items() if c >= min_support}
    if not level:
        return []

    frequent_by_len: list[dict[frozenset[int], int]] = [level]
    while level and len(next(iter(level))) < max_len:
        cands: set[frozenset[int]] = set()
        keys = list(level)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                u = keys[a] | keys[b]
                if len(u) == len(keys[a]) + 1:
                    cands.add(u)
        nxt: dict[frozenset[int], int] = {}
        for c in cands:
            n = sum(1 for t in transactions if c <= t)
            if n >= min_support:
                nxt[c] = n
        if not nxt:
            break
        frequent_by_len.append(nxt)
        level = nxt

    # An itemset has a frequent proper superset iff it has a frequent superset
    # exactly one item larger (anti-monotonicity), so only the next level needs
    # to be consulted.
    maximal: list[tuple[frozenset[int], int]] = []
    for k, d in enumerate(frequent_by_len):
        above = frequent_by_len[k + 1] if k + 1 < len(frequent_by_len) else {}
        for s, n in d.items():
            if not any(s < other for other in above):
                maximal.append((s, n))
    return maximal
